"""IceCube I3InferenceModule.

Contains functionality for writing model predictions to i3 files.
"""

from typing import List, Union, Optional, TYPE_CHECKING, Dict, Any

import numpy as np
from torch_geometric.data import Data, Batch

from graphnet.utilities.config import ModelConfig
from graphnet.deployment import DeploymentModule
from graphnet.data.extractors.icecube import I3FeatureExtractor
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
    )  # pyright: reportMissingImports=false


class I3InferenceModule(DeploymentModule):
    """General class for inference on i3 frames."""

    def __init__(
        self,
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model_config: Union[ModelConfig, str],
        state_dict: str,
        model_name: str,
        gcd_file: str,
        features: Optional[List[str]] = None,
        prediction_columns: Optional[Union[List[str], None]] = None,
        pulsemap: Optional[str] = None,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model_config: The ModelConfig (or path to it) that summarizes the
                            model used for inference.
            state_dict: Path to state_dict containing the learned weights.
            model_name: The name used for the model. Will help define the
                        named entry in the I3Frame. E.g. "dynedge".
            gcd_file: path to associated gcd file.
            features: the features of the pulsemap that the model is expecting.
            prediction_columns: column names for the predictions of the model.
                               Will help define the named entry in the I3Frame.
                                E.g. ['energy_reco']. Optional.
            pulsemap: the pulsmap that the model is expecting as input.
        """
        super().__init__(
            model_config=model_config,
            state_dict=state_dict,
            prediction_columns=prediction_columns,
        )
        # Checks
        assert isinstance(gcd_file, str), "gcd_file must be string"

        # Set Member Variables
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]
        if features is None:
            features = self.model._graph_definition._input_feature_names
        self._graph_definition = self.model._graph_definition
        self._pulsemap = pulsemap
        self._gcd_file = gcd_file
        self.model_name = model_name
        self._features = features

        # Set GCD file for pulsemap extractor
        for i3_extractor in self._i3_extractors:
            i3_extractor.set_gcd(i3_file="", gcd_file=self._gcd_file)

    def __call__(self, frame_cache: [I3Frame]) -> bool:
        """Write predictions from model to frame."""
        # inference
        data, valid_mask = self._create_data_representation(frame_cache=frame_cache)
        predictions = self._apply_model(data=data)

        ### reinsert dropped predictions
        full_predictions = []
        pred_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                full_predictions.append(predictions[pred_idx])
                pred_idx += 1
            else:
                full_predictions.append(np.full(len(self.prediction_columns), np.nan))
        predictions = np.vstack(full_predictions)

        # Check dimensions of predictions and prediction columns
        dim = self._check_dimensions(predictions=predictions)

        # Build Dictionary from predictions
        data = self._create_dictionary(dim=dim, predictions=predictions)

        # Submit Dictionary to frame
        frame_cache = self._add_to_frame(frame_cache=frame_cache, data_list=data)
        return True

    def _check_dimensions(self, predictions: np.ndarray) -> int:
        if len(predictions.shape) > 1:
            dim = predictions.shape[1]
        else:
            dim = len(predictions)
        try:
            assert dim == len(self.prediction_columns)
        except AssertionError as e:
            self.error(
                f"predictions have shape {dim} but"
                f"prediction columns have [{self.prediction_columns}]"
            )
            raise e
        return dim

    def _create_dictionary(
        self, dim: int, predictions: np.ndarray
    ) -> [Dict[str, Any]]:
        """Transform predictions into a dictionary."""
        all_data = []
        for row in predictions:
            data = {}
            for i in range(dim):
                data[self.model_name + "_" + self.prediction_columns[i]] = I3Double(float(row[i]))
            all_data.append(data)
        return all_data

    def _apply_model(self, data: Data) -> np.ndarray:
        """Apply model to `Data` and case-handling."""
        if data is not None:
            predictions = self._inference(data)
            if isinstance(predictions, list):
                predictions = predictions[0]
                self.warning(
                    f"{self.__class__.__name__} assumes one Task "
                    f"but got {len(predictions)}. Only the first will"
                    " be used."
                )
        else:
            self.warning(
                "At least one event has no pulses "
                " - padding {self.prediction_columns} with NaN."
            )
            predictions = np.repeat(
                [np.nan], len(self.prediction_columns)
            ).reshape(-1, len(self.prediction_columns))
        return predictions

    def _create_data_representation(self, frame_cache: [I3Frame]) -> Data:
        """Process Physics I3Frame into graph."""
        data_cache = []
        valid_mask = []
        for f in frame_cache:
            # Extract features
            input_features = self._extract_feature_array_from_frame(f)
            # Prepare graph data
            if input_features is not None and len(input_features) > 0 :
                data = self._graph_definition(
                    input_features=input_features,
                    input_feature_names=self._features,
                )
                data_cache.append(data)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        if data_cache:
            batch = Batch.from_data_list(data_cache)
        else:
            batch = None
        return batch, valid_mask

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        """Apply the I3FeatureExtractors to the I3Frame.

        Arguments:
            frame: Physics I3Frame (PFrame)

        Returns:
            array with pulsemap
        """
        features = None
        for i3extractor in self._i3_extractors:
            feature_dict = i3extractor(frame)
            if feature_dict is None:
                continue
            features_pulsemap = np.array(
                [feature_dict[key] for key in self._features]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _add_to_frame(self, frame_cache: [I3Frame], data_list: [Dict[str, Any]]) -> I3Frame:
        """Add every field in data to I3Frame.

        Arguments:
            frame: I3Frame (physics)
            data: Dictionary containing content that will be written to frame.

        Returns:
            frame: Same I3Frame as input, but with the new entries
        """
        assert len(frame_cache) == len(data_list), (
            f"Number of frames ({len(frame_cache)}) and data dictionaries ({len(data_list)}) must match"
        )
        for frame, data in zip(frame_cache, data_list):
            assert isinstance(data, dict), f"Each data item must be dict, got {type(data)}"
            for key, value in data.items():
                if key not in frame:
                    frame.Put(key, value)
        return frame_cache
