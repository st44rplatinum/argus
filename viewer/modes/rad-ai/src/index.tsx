import {
  ohif,
  cornerstone,
  extensionDependencies as basicExtensionDependencies,
  layoutTemplate,
  modeFactory,
} from '@ohif/mode-basic';

import { id } from './id';

// extensionDependencies is an OBJECT of packageName -> semver range, not an
// array. Spreading it into `[...]` throws "not iterable" before the mode ever
// loads. Same spread, object literal.
const extensionDependencies = {
  ...basicExtensionDependencies,
  '@argus/extension-rad-ai': '^0.1.0',
};

// Panel ids are <packageName>.panelModule.<name>, where <name> comes from the
// object returned by the extension's getPanelModule.
const seriesInfoPanel = '@argus/extension-rad-ai.panelModule.seriesInfo';

const modeInstance = {
  id,
  routeName: 'rad-ai',
  displayName: 'Rad AI',

  // The protocol registered by extensions/cornerstone/src/hps/mpr.ts. Plain
  // string is fine here — basic uses 'default', segmentation uses an array.
  hangingProtocol: 'mpr',

  routes: [
    {
      path: 'rad-ai',

      layoutTemplate,

      layoutInstance: {
        // layoutTemplate() clones layoutInstance and hands it back. Without an
        // id there is no template to instantiate and the route renders nothing
        // — a blank screen with no error, which is the worst kind.
        id: ohif.layout,

        props: {
          // Without this there is no series list, so no way to change series.
          leftPanels: [ohif.thumbnailList],
          leftPanelResizable: true,

          rightPanels: [seriesInfoPanel],
          rightPanelResizable: true,

          viewports: [
            {
              // `cornerstone` is an object of module ids, not an id itself.
              namespace: cornerstone.viewport,

              // An array of SOP-class-handler id STRINGS. It answers "which
              // kinds of display set can this viewport show", not "how should
              // it show them" — viewport type comes from the hanging protocol.
              displaySetsToDisplay: [ohif.sopClassHandler],
            },
          ],
        },
      },
    },
  ],

  extensions: extensionDependencies,

  // Nothing is displayable unless its handler is listed here as well.
  sopClassHandlers: [ohif.sopClassHandler],
};

export default {
  id,
  modeFactory,
  modeInstance,
  extensionDependencies,
};
