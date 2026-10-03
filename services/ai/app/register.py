from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

@dataclass
class RegistrationResult:
    transform: sitk.Transform
    rigid: sitk.Euler3DTransform
    bspline: sitk.BSplineTransform | None
    metric_value: float

def body_mask(image: sitk.Image) -> sitk.Image:
    """
    Remove the table and external air from the
    registration metric
    """

    threshold = sitk.BinaryThreshold(
        image,
        lowerThreshold=-500.0,
        upperThreshold=5000.0,
        insideValue=1, 
        outsideValue=0,
    )

    threshold = sitk.Cast(threshold, sitk.sitkUInt8)

    cc = sitk.ConnectedComponent(threshold)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)

    largest = sitk.BinaryThreshold(
        relabeled, 
        lowerThreshold=1,
        upperThreshold=1,
        insideValue=1,
        outsideValue=0,
    )

    return sitk.Cast(largest, sitk.sitkUInt8)

def _configure_metric(registration, fixed):
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50,
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM,
    )

    registration.SetMetricSamplingPercentage(
        0.03,
    )

    registration.SetMetricFixedMask(body_mask(fixed))
    registration.SetInterpolator(sitk.sitkLinear)

def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
) -> RegistrationResult:
    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()

    _configure_metric(registration, fixed)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.01,
        numberOfIterations=150,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-4,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(transform, inPlace=False)

    final = registration.Execute(fixed, moving)

    return RegistrationResult(
        transform=final,
        rigid=final,
        bspline=None,
        metric_value=float(registration.GetMetricValue()),
    )

def register_deformable(
    fixed: sitk.Image,
    moving: sitk.Image,
    rigid: sitk.Transform,
) -> RegistrationResult:
    # Work at approximately 3 mm.
    spacing = np.array(fixed.GetSpacing(), dtype=float)
    size = np.array(fixed.GetSize(), dtype=int)

    target_spacing = 3.0
    shrink = np.maximum(
        np.rint(target_spacing / spacing).astype(int),
        1,
    )

    shrink = [int(v) for v in shrink]

    fixed_small = sitk.Shrink(fixed, shrink)
    moving_small = sitk.Shrink(moving, shrink)

    # ~50 mm control-point spacing.
    physical_size = np.array(fixed_small.GetSize()) * np.array(
        fixed_small.GetSpacing()
    )
    mesh = np.maximum(
        np.rint(physical_size / 50.0).astype(int),
        1,
    )

    bspline = sitk.BSplineTransformInitializer(
        fixed_small,
        [int(v) for v in mesh],
        order=3,
    )

    registration = sitk.ImageRegistrationMethod()

    _configure_metric(registration, fixed_small)

    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=80,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])

    # First put the moving image into rigid alignment.
    rigid_small = sitk.Resample(
        moving_small,
        fixed_small,
        rigid,
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )

    registration.SetInitialTransform(bspline, inPlace=False)

    final_bspline = registration.Execute(
        fixed_small,
        rigid_small,
    )

    # B-spline was optimised in the 3 mm domain. The transform itself is
    # physical-space based, so it can be evaluated against full resolution.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid)
    composite.AddTransform(final_bspline)
    # the rigid result is itself a composite; HDF5 cannot save one nested
    composite.FlattenTransform()

    return RegistrationResult(
        transform=composite,
        rigid=rigid,
        bspline=final_bspline,
        metric_value=float(registration.GetMetricValue()),
    )

def register(
        fixed: sitk.Image,
        moving: sitk.Image,
) -> RegistrationResult:
    rigid = register_rigid(fixed, moving)

    return register_deformable(
        fixed, moving,
        rigid.transform,
    )

def transform_point(
        transform: sitk.Transform,
        point,
):
    return tuple(
        float(v)
        for v in transform.TransformPoint(tuple(map(float, point)))
    )

def no_registration():
    return sitk.Transform(3, sitk.sitkIdentity)

def tre(
    transform: sitk.Transform,
    fixed_point,
    moving_point,
) -> float:
    predicted = np.asarray(
        transform.TransformPoint(tuple(fixed_point)),
        dtype=float,
    )
    expected = np.asarray(moving_point, dtype=float)

    return float(np.linalg.norm(predicted - expected))


def tre_table(
    transforms: dict[str, sitk.Transform],
    landmarks: dict[str, tuple],
):
    rows = []

    for transform_name, transform in transforms.items():
        for landmark_name, points in landmarks.items():
            error = tre(
                transform,
                points["fixed"],
                points["moving"],
            )

            rows.append({
                "transform": transform_name,
                "landmark": landmark_name,
                "tre_mm": error,
            })

    return rows
