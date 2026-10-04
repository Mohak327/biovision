// The circular viewport: the original on the left of a draggable divider,
// the reconstruction on the right.
import { type KeyboardEvent, type PointerEvent, useRef, useState } from "react";

type Props = {
  original: string | null;
  reconstruction: string | null;
  speciesName: string;
  busy: boolean;
};

export function Eyepiece({ original, reconstruction, speciesName, busy }: Props) {
  const [split, setSplit] = useState(50); // percent of the width showing the original
  const well = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const moveTo = (clientX: number) => {
    const box = well.current?.getBoundingClientRect();
    if (!box) return;
    setSplit(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)));
  };
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    moveTo(event.clientX);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) moveTo(event.clientX);
  };
  const onPointerUp = () => {
    dragging.current = false;
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 2;
    if (event.key === "ArrowLeft") setSplit((value) => Math.max(0, value - step));
    else if (event.key === "ArrowRight") setSplit((value) => Math.min(100, value + step));
    else if (event.key === "Home") setSplit(0);
    else if (event.key === "End") setSplit(100);
    else return;
    event.preventDefault();
  };

  return (
    <div className="eyepiece">
      <div
        className="eyepiece-well"
        ref={well}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        aria-busy={busy}
      >
        {reconstruction ? (
          <img className="eyepiece-image" src={reconstruction} alt={`The picture rebuilt from the ${speciesName}'s spikes`} draggable={false} />
        ) : (
          <div className="eyepiece-empty">Waiting for the first spikes</div>
        )}
        {original && (
          <img
            className="eyepiece-image"
            src={original}
            alt="The original picture"
            draggable={false}
            style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
          />
        )}
        <div
          className="eyepiece-divider"
          style={{ left: `${split}%` }}
          role="slider"
          tabIndex={0}
          aria-label="Divider between the original and the rebuilt picture"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          aria-valuetext={`${Math.round(split)} percent original`}
          onKeyDown={onKeyDown}
        >
          <span className="eyepiece-grip" aria-hidden="true" />
        </div>
      </div>
      <div className="eyepiece-captions" aria-hidden="true">
        <span>the world</span>
        <span>what the {speciesName} keeps</span>
      </div>
    </div>
  );
}
