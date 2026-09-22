/**
 * OHIF configuration for the local Orthanc.
 *
 * Derived from the viewer's own `local_orthanc.js`, with three changes:
 * ports match our compose file, the DCM4CHEE name is corrected, and the demo
 * error handler that redirects to ohif.org is gone.
 *
 * This file is ours and lives in the repo. `scripts/link_viewer.py` copies it
 * into the OHIF checkout, which is gitignored.
 *
 * The roots below are relative on purpose. The dev server proxies
 * /pacs/dicom-web to http://localhost:8042/dicom-web, so the browser only ever
 * talks to its own origin. That is what lets Orthanc run with CORS off
 * entirely — see docker-compose.yml. Point these at :8042 directly and you are
 * back to needing a CORS wildcard on an unauthenticated PACS, which is worse
 * than it sounds.
 *
 * Start it with `python scripts/dev_viewer.py`, which sets the proxy
 * environment these paths assume.
 *
 * @type {AppTypes.Config}
 */
window.config = {
  routerBasename: null,
  extensions: [],
  modes: [],
  showStudyList: true,
  // Four workers on a twelve-core box leaves room for the browser itself.
  maxNumberOfWebWorkers: 4,
  showLoadingIndicator: true,
  showWarningMessageForCrossOrigin: true,
  // The GTX 1050 Ti will fall back to CPU on some volume operations. Say so
  // rather than appearing to hang.
  showCPUFallbackMessage: true,
  strictZSpacingForVolumeViewport: true,
  defaultDataSourceName: 'orthanc',
  dataSources: [
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'orthanc',
      configuration: {
        friendlyName: 'Local Orthanc',
        name: 'Orthanc',
        // Relative — resolved against the dev server, then proxied. See above.
        wadoUriRoot: '/pacs/dicom-web',
        qidoRoot: '/pacs/dicom-web',
        wadoRoot: '/pacs/dicom-web',
        // Verified against this Orthanc in P0 — see scripts/check_dicomweb.py,
        // which relies on includefield to read StudyDescription.
        qidoSupportsIncludeField: true,
        supportsReject: true,
        dicomUploadEnabled: true,
        imageRendering: 'wadors',
        thumbnailRendering: 'rendered',
        thumbnailRequestStrategy: 'fetch',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: true,
        supportsWildcard: true,
        omitQuotationForMultipartRequest: true,
        bulkDataURI: {
          enabled: true,
        },
      },
    },
    {
      // Useful for opening a file without pushing it to the PACS first.
      namespace: '@ohif/extension-default.dataSourcesModule.dicomlocal',
      sourceName: 'dicomlocal',
      configuration: {
        friendlyName: 'Open local file',
      },
    },
  ],
};
