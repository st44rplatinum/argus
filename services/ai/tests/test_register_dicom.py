"""The registered series lands each slice on the right instance."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset

from app.register_dicom import store_registered_series


class FakeClient:
    def __init__(self):
        self.stored = []

    def store_instances(self, instances):
        self.stored += instances


def test_slices_follow_volume_order_not_server_order():
    order = [f"1.2.3.{i}" for i in range(6)]
    sources = []
    for uid in reversed(order):  # the server hands them back in some other order
        ds = Dataset()
        ds.SOPInstanceUID = uid
        ds.FrameOfReferenceUID = "9.9"
        sources.append(ds)

    # slice k is filled with the value k
    array = np.repeat(np.arange(6, dtype=np.float32)[:, None, None], 4, axis=1).repeat(4, axis=2)
    client = FakeClient()
    store_registered_series(client, sources, order, sitk.GetImageFromArray(array), "Prior, registered")

    for k, (ds, uid) in enumerate(zip(client.stored, order)):
        assert np.frombuffer(ds.PixelData, np.int16)[0] == k
        assert ds.FrameOfReferenceUID == "9.9"   # geometry kept
        assert ds.SOPInstanceUID != uid          # but a new instance
    assert len({ds.SeriesInstanceUID for ds in client.stored}) == 1
