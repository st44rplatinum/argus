import React, { useCallback, useEffect, useState } from 'react';

import { LesionChange, RegisterResult } from '../api';
import useJob from '../useJob';

type Current = {
  studyUid: string;
  seriesUid: string;
  patientId: string;
  studyDate: string;
  frameOfReference: string;
};

type Prior = { studyUid: string; seriesUid: string; date: string; description: string };

const prettyDate = (d: string) => (d?.length === 8 ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6)}` : d);
const ml = (mm3: number) => `${(mm3 / 1000).toFixed(2)} ml`;
const percent = (p: number | null) => (p === null ? '–' : `${p > 0 ? '+' : ''}${p.toFixed(0)}%`);

export default function LongitudinalPanel({ servicesManager, extensionManager }: withAppTypes) {
  const { viewportGridService, displaySetService, measurementService } = servicesManager.services;

  const [current, setCurrent] = useState<Current | null>(null);
  const [priors, setPriors] = useState<Prior[]>([]);
  const [priorUid, setPriorUid] = useState('');
  const [probe, setProbe] = useState<number[] | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const register = useJob();
  const lesion = useJob();

  // The CT in the active viewport, or nothing.
  const readCurrent = useCallback(() => {
    const { activeViewportId, viewports } = viewportGridService.getState();
    const uid = viewports?.get(activeViewportId)?.displaySetInstanceUIDs?.[0];
    const ds = uid ? displaySetService.getDisplaySetByUID(uid) : null;
    const instance = ds?.instance ?? ds?.instances?.[0];

    if (!ds || ds.Modality !== 'CT' || !instance) {
      setCurrent(null);
      return;
    }
    setCurrent({
      studyUid: ds.StudyInstanceUID,
      seriesUid: ds.SeriesInstanceUID,
      patientId: instance.PatientID,
      studyDate: instance.StudyDate,
      frameOfReference: instance.FrameOfReferenceUID,
    });
  }, [viewportGridService, displaySetService]);

  useEffect(() => {
    readCurrent();
    const subs = [
      viewportGridService.subscribe(viewportGridService.EVENTS.GRID_STATE_CHANGED, readCurrent),
      viewportGridService.subscribe(viewportGridService.EVENTS.ACTIVE_VIEWPORT_ID_CHANGED, readCurrent),
    ];
    return () => subs.forEach(s => s.unsubscribe());
  }, [readCurrent, viewportGridService]);

  // Older studies of the same patient, each with its CT series.
  useEffect(() => {
    if (!current) {
      return;
    }
    let cancelled = false;

    (async () => {
      try {
        const [dataSource] = extensionManager.getActiveDataSource();
        const studies = await dataSource.query.studies.search({ patientId: current.patientId });

        const found: Prior[] = [];
        // the search wildcards the id, so FU-s0012 would also find FU-s0012-rigid
        for (const study of studies.filter(s => s.mrn === current.patientId)) {
          if (study.studyInstanceUid === current.studyUid || study.date >= current.studyDate) {
            continue;
          }
          const series = await dataSource.query.series.search(study.studyInstanceUid);
          const ct = series.find(s => s.modality === 'CT' && !/registered/i.test(s.description));
          if (ct) {
            found.push({
              studyUid: study.studyInstanceUid,
              seriesUid: ct.seriesInstanceUid,
              date: study.date,
              description: study.description,
            });
          }
        }

        found.sort((a, b) => b.date.localeCompare(a.date));
        if (!cancelled) {
          setPriors(found);
          setPriorUid(found[0]?.studyUid ?? '');
        }
      } catch (exc) {
        if (!cancelled) {
          setMessage(String(exc));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [current?.studyUid, current?.patientId, extensionManager]);

  // The newest Probe on this scan is the lesion to measure.
  const readProbe = useCallback(() => {
    if (!current) {
      setProbe(null);
      return;
    }
    const probes = measurementService
      .getMeasurements()
      .filter(m => m.toolName === 'Probe' && m.FrameOfReferenceUID === current.frameOfReference);
    const last = probes[probes.length - 1];
    setProbe(last?.points?.[0] ? Array.from(last.points[0]) : null);
  }, [current, measurementService]);

  useEffect(() => {
    readProbe();
    const { MEASUREMENT_ADDED, MEASUREMENT_UPDATED, MEASUREMENT_REMOVED } = measurementService.EVENTS;
    const subs = [MEASUREMENT_ADDED, MEASUREMENT_UPDATED, MEASUREMENT_REMOVED].map(e =>
      measurementService.subscribe(e, readProbe)
    );
    return () => subs.forEach(s => s.unsubscribe());
  }, [readProbe, measurementService]);

  const prior = priors.find(p => p.studyUid === priorUid);

  const base = () => ({
    study_uid: current.studyUid,
    series_uid: current.seriesUid,
    prior_study_uid: prior.studyUid,
    prior_series_uid: prior.seriesUid,
    prior_study_date: prettyDate(prior.date),
  });

  const registered = register.job?.status === 'done' ? (register.job.result as RegisterResult) : null;
  const change = lesion.job?.status === 'done' ? (lesion.job.result as LesionChange) : null;

  if (!current) {
    return <p className="text-muted-foreground p-3 text-xs">Select a CT series.</p>;
  }

  return (
    <div className="text-foreground p-3 text-xs">
      <p className="text-muted-foreground">
        {current.patientId} · this scan {prettyDate(current.studyDate)}
      </p>

      <label className="text-muted-foreground mt-3 block">Prior</label>
      {priors.length === 0 ? (
        <p className="mt-1">No earlier studies for this patient.</p>
      ) : (
        <select
          className="bg-background text-foreground border-border mt-1 w-full rounded border px-2 py-1"
          value={priorUid}
          onChange={e => setPriorUid(e.target.value)}
          disabled={register.running}
        >
          {priors.map(p => (
            <option
              key={p.studyUid}
              value={p.studyUid}
            >
              {prettyDate(p.date)} · {p.description || 'no description'}
            </option>
          ))}
        </select>
      )}

      <button
        className="bg-primary text-primary-foreground mt-3 w-full rounded px-3 py-2 font-medium disabled:opacity-50"
        onClick={() => register.submit({ ...base(), model: 'register' })}
        disabled={!prior || register.running}
      >
        {register.running ? `Registering… ${register.job?.stage ?? ''}` : 'Register prior'}
      </button>

      {register.error ? <p className="mt-2 break-words text-red-500">{register.error}</p> : null}

      {registered ? (
        <p className="text-muted-foreground mt-2">
          Stored as “Prior {prior ? prettyDate(prior.date) : ''}, registered”
          {register.job?.cached ? ' (cached)' : ` in ${registered.seconds.toFixed(0)} s`}. Reload,
          put it beside this series, and turn on Image Slice Sync under More tools.
        </p>
      ) : null}

      <div className="border-border mt-4 border-t pt-3">
        <p className="text-muted-foreground">
          {probe
            ? 'Probe placed on the lesion.'
            : 'Drop a Probe (More tools) on a lesion in this scan to measure it.'}
        </p>

        <button
          className="bg-primary text-primary-foreground mt-2 w-full rounded px-3 py-2 font-medium disabled:opacity-50"
          onClick={() => lesion.submit({ ...base(), model: 'lesion_change', point: probe })}
          disabled={!prior || !probe || !registered || lesion.running}
          title={registered ? '' : 'Register the prior first'}
        >
          {lesion.running ? `Measuring… ${lesion.job?.stage ?? ''}` : 'Measure change'}
        </button>

        {lesion.error ? <p className="mt-2 break-words text-red-500">{lesion.error}</p> : null}

        {change ? (
          <table className="mt-3 w-full">
            <thead className="text-muted-foreground">
              <tr>
                <th className="text-left font-normal"></th>
                <th className="text-right font-normal">volume</th>
                <th className="text-right font-normal">diameter</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              <tr>
                <td className="font-sans">prior</td>
                <td className="text-right">{ml(change.prior.volume_mm3)}</td>
                <td className="text-right">{change.prior.diameter_mm.toFixed(1)} mm</td>
              </tr>
              <tr>
                <td className="font-sans">now</td>
                <td className="text-right">{ml(change.current.volume_mm3)}</td>
                <td className="text-right">{change.current.diameter_mm.toFixed(1)} mm</td>
              </tr>
              <tr className="font-medium">
                <td className="font-sans">change</td>
                <td className="text-right">{percent(change.change_percent)}</td>
                <td className="text-right">
                  {percent(
                    change.prior.diameter_mm > 0
                      ? ((change.current.diameter_mm - change.prior.diameter_mm) / change.prior.diameter_mm) * 100
                      : null
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        ) : null}

        {change ? (
          <p className="text-muted-foreground mt-2">
            Measured in each scan's own pixels. In the warped prior it would read{' '}
            {percent(change.warped_change_percent)}.
          </p>
        ) : null}
      </div>

      {message ? <p className="mt-3 break-words text-red-500">{message}</p> : null}
    </div>
  );
}
