import React from 'react';

import SeriesInfoPanel from './panels/SeriesInfoPanel';

export default function getPanelModule({ servicesManager }: withAppTypes) {
  return [
    {
      name: 'seriesInfo',
      iconName: 'tab-studies',
      iconLabel: 'AI',
      label: 'Series',
      component: () => <SeriesInfoPanel servicesManager={servicesManager} />,
    },
  ];
}
