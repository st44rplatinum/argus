import { Types } from '@ohif/core';

import { id } from './id';
import getPanelModule from './getPanelModule';

/**
 * Everything of ours that hangs in the viewer.
 *
 * This is the analogue of eov's plugin ABI: the viewer stays somebody else's
 * code, and our work attaches through one declared surface. Keeping that line
 * sharp is what lets the OHIF checkout stay disposable.
 *
 * Panels here, commands and AI job handling as P3 lands.
 */
const radAiExtension: Types.Extensions.Extension = {
  id,
  getPanelModule,
};

export default radAiExtension;
export { id };
