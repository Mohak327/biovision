"""An ordered, validated list of stages that encodes an image into a neural code."""
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .field import VisualField
from .stage import LinearStage, PointwiseStage, Stage


@dataclass(frozen=True)
class NeuralCode:
    """The output of a pipeline, plus every stage's output for plotting."""

    responses: np.ndarray
    intermediates: dict[str, np.ndarray]
    pipeline_name: str


@dataclass(frozen=True)
class Pipeline:
    name: str
    field: VisualField
    stages: tuple[Stage, ...]
    description: str = ""
    citations: tuple[str, ...] = ()
    metadata: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")
        seen_pointwise = False
        for stage in self.stages:
            if isinstance(stage, PointwiseStage):
                seen_pointwise = True
            elif isinstance(stage, LinearStage):
                if seen_pointwise:
                    raise ValueError(
                        f"linear stage '{stage.name}' follows a pointwise stage; "
                        "all linear stages must come first"
                    )
            else:
                raise ValueError(f"{stage!r} is not a LinearStage or PointwiseStage")
        if not self.linear_stages:
            raise ValueError("a pipeline needs at least one linear stage")
        for a, b in zip(self.linear_stages, self.linear_stages[1:]):
            if a.out_shape != b.in_shape:
                raise ValueError(
                    f"shape mismatch: '{a.name}' outputs {a.out_shape} "
                    f"but '{b.name}' expects {b.in_shape}"
                )

    @property
    def linear_stages(self) -> tuple[LinearStage, ...]:
        return tuple(s for s in self.stages if isinstance(s, LinearStage))

    @property
    def pointwise_stages(self) -> tuple[PointwiseStage, ...]:
        return tuple(s for s in self.stages if isinstance(s, PointwiseStage))

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self.linear_stages[0].in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self.linear_stages[-1].out_shape

    @property
    def n_neurons(self) -> int:
        return int(np.prod(self.out_shape))

    def replace(self, stage: Stage) -> "Pipeline":
        """A copy with the stage of the same name swapped for `stage`."""
        if stage.name not in [s.name for s in self.stages]:
            raise KeyError(f"no stage named '{stage.name}' in pipeline '{self.name}'")
        stages = tuple(stage if s.name == stage.name else s for s in self.stages)
        return Pipeline(self.name, self.field, stages, self.description,
                        self.citations, self.metadata)

    def encode(self, image: np.ndarray, rng: np.random.Generator | None = None) -> NeuralCode:
        """Run `image` (channels, size, size) through every stage."""
        image = np.asarray(image, dtype=float)
        if image.shape != self.in_shape:
            raise ValueError(f"expected image of shape {self.in_shape}, got {image.shape}")
        x = image
        intermediates = {}
        for stage in self.linear_stages:
            x = stage.encode(x, rng)
            intermediates[stage.name] = x
        for stage in self.pointwise_stages:
            x = stage.forward(x, rng)
            intermediates[stage.name] = x
        return NeuralCode(x, intermediates, self.name)

    def linear_operator(self) -> LinearOperator:
        """All linear stages composed into one operator A on flattened arrays."""
        stages = self.linear_stages

        def matvec(v):
            x = v.reshape(self.in_shape)
            for stage in stages:
                x = stage.forward(x)
            return x.ravel()

        def rmatvec(v):
            y = v.reshape(self.out_shape)
            for stage in reversed(stages):
                y = stage.adjoint(y)
            return y.ravel()

        shape = (self.n_neurons, int(np.prod(self.in_shape)))
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
