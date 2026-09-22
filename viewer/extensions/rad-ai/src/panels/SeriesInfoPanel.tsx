import React, { useEffect, useState } from 'react';

/**
 * What the AI service will be asked about, shown before anything is asked.
 *
 * This panel is deliberately not a placeholder. It reports the identifiers and
 * geometry of whatever series is in the active viewport — the same values
 * services/ai/app/dicomweb.py needs to fetch and assemble a volume. When a
 * segmentation later lands in the wrong place, this is where you find out
 * whether the viewer and the service disagreed about the series.
 */

type Row = { label: string; value: string };

function Field({ label, value }: Row) {
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

function describe(displaySet): Row[] {
  if (!displaySet) {
    return [];
  }

  const images = displaySet.images ?? [];
  const first = images[0];

  const rows: Row[] = [
    { label: 'Modality', value: displaySet.Modality ?? '-' },
    { label: 'Series', value: displaySet.SeriesDescription ?? '-' },
    { label: 'Instances', value: String(images.length || displaySet.numImageFrames || 0) },
  ];

  if (first) {
    const [rowSpacing, columnSpacing] = first.PixelSpacing ?? [];
    rows.push(
      { label: 'Size', value: `${first.Columns ?? '?'} x ${first.Rows ?? '?'}` },
      {
        label: 'Pixel spacing',
        value:
          rowSpacing && columnSpacing
            ? `${Number(rowSpacing).toFixed(2)} x ${Number(columnSpacing).toFixed(2)} mm`
            : 'absent',
      },
      {
        label: 'Slice thickness',
        value: first.SliceThickness ? `${Number(first.SliceThickness).toFixed(2)} mm` : 'absent',
      }
    );
  }

  rows.push(
    { label: 'StudyInstanceUID', value: displaySet.StudyInstanceUID ?? '-' },
    { label: 'SeriesInstanceUID', value: displaySet.SeriesInstanceUID ?? '-' }
  );

  return rows;
}

export default function SeriesInfoPanel({ servicesManager }: withAppTypes) {
  const { viewportGridService, displaySetService } = servicesManager.services;
  const [displaySet, setDisplaySet] = useState(null);

  useEffect(() => {
    const read = () => {
      const { activeViewportId, viewports } = viewportGridService.getState();
      const viewport = viewports.get(activeViewportId);
      const uid = viewport?.displaySetInstanceUIDs?.[0];
      setDisplaySet(uid ? displaySetService.getDisplaySetByUID(uid) : null);
    };

    read();

    // Both matter: the user can switch viewports, and a viewport's display set
    // can change under them via a hanging protocol.
    const subscriptions = [
      viewportGridService.subscribe(
        viewportGridService.EVENTS.ACTIVE_VIEWPORT_ID_CHANGED,
        read
      ),
      viewportGridService.subscribe(viewportGridService.EVENTS.GRID_STATE_CHANGED, read),
    ];

    return () => subscriptions.forEach(subscription => subscription.unsubscribe());
  }, [viewportGridService, displaySetService]);

  const rows = describe(displaySet);

  return (
    <div className="p-2">
      {rows.length === 0 ? (
        <p className="text-muted-foreground p-2 text-xs">No series in the active viewport.</p>
      ) : (
        <div className="divide-border divide-y">
          {rows.map(row => (
            <Field
              key={row.label}
              label={row.label}
              value={row.value}
            />
          ))}
        </div>
      )}
    </div>
  );
}
