import React from 'react';

import JobsPanel from './panels/JobsPanel';
import SeriesInfoPanel from './panels/SeriesInfoPanel';
import SimilarPanel from './panels/SimilarPanel';

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
    {
      name: 'similar',
      iconName: 'tab-linear',
      iconLabel: 'Similar',
      label: 'Similar cases',
      component: () => <SimilarPanel servicesManager={servicesManager} />,
    },
  ];
}
