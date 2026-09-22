import React, { useEffect, useState } from 'react';

/**
 * What the AI service will be asked about, shown before anything is asked.
 *
 * Not a placeholder: these are the values services/ai/app/dicomweb.py needs to
 * fetch and assemble a volume. Spacing in particular is computed here the same
 * way the service computes it — from successive slice positions rather than
 * SliceThickness. When a mask later lands in the wrong place, two independent
 * numbers that agree tell you the geometry was not the problem.
 */

type SeriesInfo = {
  studyInstanceUID: string;
  seriesInstanceUID: string;
  modality: string;
  numberOfSlices: number;
  sliceSpacing: number | null;
  ohifSpacing: number | null;
};

const EMPTY: SeriesInfo = {
  studyInstanceUID: '',
  seriesInstanceUID: '',
  modality: '',
  numberOfSlices: 0,
  sliceSpacing: null,
  ohifSpacing: null,
};

/** Mean distance between successive slice positions, in millimetres. */
function calculateSliceSpacing(instances: any[]): number | null {
  if (!instances || instances.length < 2) {
    return null;
  }

  const positions = instances
    .map(instance => instance.ImagePositionPatient)
    .filter(position => position && position.length >= 3)
    .map(position => [Number(position[0]), Number(position[1]), Number(position[2])]);

  if (positions.length < 2) {
    return null;
  }

  const distances: number[] = [];

  for (let i = 1; i < positions.length; i++) {
    const previous = positions[i - 1];
    const current = positions[i];

    const distance = Math.sqrt(
      Math.pow(current[0] - previous[0], 2) +
        Math.pow(current[1] - previous[1], 2) +
        Math.pow(current[2] - previous[2], 2)
    );

    if (Number.isFinite(distance)) {
      distances.push(distance);
    }
  }

  if (!distances.length) {
    return null;
  }

  return distances.reduce((sum, value) => sum + value, 0) / distances.length;
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2 py-1 text-xs">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span
        className="text-foreground truncate text-right font-mono"
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

export default function SeriesInfoPanel({ servicesManager }: withAppTypes) {
  const { viewportGridService, displaySetService } = servicesManager.services;
  const [info, setInfo] = useState<SeriesInfo>(EMPTY);

  useEffect(() => {
    const read = () => {
      const { activeViewportId, viewports } = viewportGridService.getState();
      const viewport = viewports?.get(activeViewportId);
      const displaySetUID = viewport?.displaySetInstanceUIDs?.[0];

      if (!displaySetUID) {
        setInfo(EMPTY);
        return;
      }

      const displaySet = displaySetService.getDisplaySetByUID(displaySetUID);

      if (!displaySet) {
        setInfo(EMPTY);
        return;
      }

      // ImageSet sets `instances = images`, so these are the same array.
      const instances = displaySet.instances ?? displaySet.images ?? [];

      setInfo({
        studyInstanceUID: displaySet.StudyInstanceUID ?? '',
        seriesInstanceUID: displaySet.SeriesInstanceUID ?? '',
        modality: displaySet.Modality ?? '',
        numberOfSlices: instances.length,
        sliceSpacing: calculateSliceSpacing(instances),
        // OHIF computes this too. Showing both is the point — if they
        // disagree, the series is not the uniform stack it appears to be.
        ohifSpacing: displaySet.averageSpacingBetweenFrames ?? null,
      });
    };

    read();

    // The active viewport can change, and the display set inside a viewport
    // can change under a hanging protocol. Both need to re-read.
    const subscriptions = [
      viewportGridService.subscribe(viewportGridService.EVENTS.ACTIVE_VIEWPORT_ID_CHANGED, read),
      viewportGridService.subscribe(viewportGridService.EVENTS.GRID_STATE_CHANGED, read),
    ];

    // subscribe() returns { unsubscribe }, not a function. React expects a
    // function here and will not call an object.
    return () => subscriptions.forEach(subscription => subscription.unsubscribe());
  }, [viewportGridService, displaySetService]);

  if (!info.seriesInstanceUID) {
    return <p className="text-muted-foreground p-3 text-xs">No series in the active viewport.</p>;
  }

  const millimetres = (value: number | null) => (value == null ? '—' : `${value.toFixed(3)} mm`);

  return (
    <div className="divide-border divide-y p-3">
      <Field
        label="Modality"
        value={info.modality || '—'}
      />
      <Field
        label="Slices"
        value={String(info.numberOfSlices || '—')}
      />
      <Field
        label="Spacing (ours)"
        value={millimetres(info.sliceSpacing)}
      />
      <Field
        label="Spacing (OHIF)"
        value={millimetres(info.ohifSpacing)}
      />
      <Field
        label="StudyInstanceUID"
        value={info.studyInstanceUID || '—'}
      />
      <Field
        label="SeriesInstanceUID"
        value={info.seriesInstanceUID || '—'}
      />
    </div>
  );
}
