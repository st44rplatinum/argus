import {
  ohif,
  cornerstone,
  dicomSeg,
  extensionDependencies as basicExtensionDependencies,
  initToolGroups,
  isValidMode,
  layoutTemplate,
  modeFactory,
  onModeEnter,
  onModeExit,
  NON_IMAGE_MODALITIES,
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
const jobsPanel = '@argus/extension-rad-ai.panelModule.jobs';

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

          rightPanels: [jobsPanel, seriesInfoPanel, cornerstone.labelMapSegmentationPanel],
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
            {
              namespace: dicomSeg.viewport,
              displaySetsToDisplay: [dicomSeg.sopClassHandler],
            },
          ],
        },
      },
    },
  ],

  extensions: extensionDependencies,

  // Nothing is displayable unless its handler is listed here as well.
  sopClassHandlers: [ohif.sopClassHandler, dicomSeg.sopClassHandler],

  // Toolbar and tool-group composition. onModeEnter reads these back off the
  // customization service, so omitting them leaves the toolbar empty.
  toolbarButtons: [{ $reference: 'cornerstone.toolbarButtons' }],
  toolbarSections: [{ $reference: 'cornerstone.toolbarSections' }],
  toolGroupAdditions: {
    default: [],
    mpr: [],
    volume3d: [],
  },

  // Read as `this.initToolGroups` inside onModeEnter — an instance property
  // precisely so a mode can substitute its own. Ours is basic's.
  initToolGroups,

  // Registered below and applied by the mode route as the bottom layer of the
  // mode scope. Empty, but it has to exist for customizations to target it.
  modeCustomizations: 'argusModeCustomizations',

  // onModeEnter reads `this.activatePanelTriggers`; undefined throws.
  activatePanelTriggers: [],

  /**
   * Lifecycle hooks. Omitting these is what produced
   * "mode?.onModeEnter is not a function" — the router calls them
   * unconditionally, so they are required rather than optional.
   */
  onModeEnter,
  onModeExit,

  validationTags: {
    study: [],
    series: [],
  },

  isValidMode,
  nonModeModalities: NON_IMAGE_MODALITIES,
};

export const customizations = {
  argusModeCustomizations: {},
};

export default {
  id,
  modeFactory,
  modeInstance,
  extensionDependencies,
  customizations,
};
