import React from 'react';

import JobsPanel from './panels/JobsPanel';
import LongitudinalPanel from './panels/LongitudinalPanel';
import SeriesInfoPanel from './panels/SeriesInfoPanel';
import SimilarPanel from './panels/SimilarPanel';

export default function getPanelModule({ servicesManager, extensionManager, commandsManager }: withAppTypes) {
  return [
    {
      name: 'seriesInfo',
      iconName: 'tab-studies',
      iconLabel: 'Series',
      label: 'Series',
      component: () => <SeriesInfoPanel servicesManager={servicesManager} />,
    },
    {
      name: 'jobs',
      iconName: 'icon-transferring',
      iconLabel: 'AI',
      label: 'AI',
      component: () => <JobsPanel servicesManager={servicesManager} />,
    },
    {
      name: 'similar',
      iconName: 'tab-linear',
      iconLabel: 'Similar',
      label: 'Similar cases',
      component: () => <SimilarPanel servicesManager={servicesManager} />,
    },
    {
      name: 'longitudinal',
      iconName: 'icon-link',
      iconLabel: 'Compare',
      label: 'Compare with prior',
      component: () => (
        <LongitudinalPanel
          servicesManager={servicesManager}
          extensionManager={extensionManager}
          commandsManager={commandsManager}
        />
      ),
    },
  ];
}
