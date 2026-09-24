import React from 'react';

import JobsPanel from './panels/JobsPanel';
import SeriesInfoPanel from './panels/SeriesInfoPanel';

export default function getPanelModule({ servicesManager }: withAppTypes) {
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
  ];
}
