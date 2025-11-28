"""The Coherent CAPTAIN Mills Detector."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector

class CCM(Detector):
    """CCM Detector class."""

    #geometry_table_path = os.path.join(CCM_GEOMETRY_TABLE_DIR, "ccm_v2.parquet")
    geometry_table_path = '/users/darcyn/workspaces/CCM/sources/darcyn/graph_net_vertex_reco/ccm_v3.parquet'

    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "dom_index0"
    sensor_id_column = "dom_index1"
    dom_type_column = "dom_type"

    #xyz = ["dom_x", "dom_y", "dom_z"]
    #string_id_column = "dom_index0"
    #sensor_id_column = "dom_index1"
    #dom_type_column = "dom_type"

    def feature_map(self) -> Dict[str, Callable]:
        """Normalize positional information."""
        feature_map = {
            "dom_x": self._dom_xy,
            "dom_y": self._dom_xy,
            "dom_z": self._dom_z,
            "time": self._time,
            "charge": self._charge,
            "dom_type": self._dom_type,
            "is_bad_dom": self._is_bad_dom,
        }
        return feature_map

    def _is_bad_dom(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0

    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0

    def _time(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0
        #return torch.log10(x)

