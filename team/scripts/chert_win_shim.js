// Preload for the Chert FaceTime CLI on Windows (team/scripts/chert.ps1 and chert.sh run `node -r <this> .../cli/main.js`).
// The CLI supports macOS and Linux only; on Windows three things break and all three are patched here, nothing else:
//  1. the entry point refuses process.platform === 'win32' outright -> report 'linux' to that check. Paths keep Windows
//     semantics (the `path` module picked win32 at startup, before this preload ran);
//  2. its owner-only file check (cli/config.js privateFile, cli/state.js) needs mode 0600 and uid === process.getuid():
//     Windows reports mode 0666 and has no getuid, so the CLI rejected the credentials it had just saved ("Not signed in").
//     lstatSync now reports the group/other bits cleared and process.getuid returns the uid Windows stats report (0).
//     The files stay regular, unlinked files under the user's profile; NTFS permissions protect them on Windows;
//  3. after every atomic write it fsyncs the DIRECTORY (byor/journal.js persist, cli/state.js saveObservation). Windows
//     answers EPERM, either at open or at fsync depending on the folder, the journal is declared unavailable and `chert dev`
//     stops. A directory open that fails hands back a placeholder descriptor; an fsync/fdatasync that fails with EPERM on a
//     descriptor that is a directory is ignored. NTFS commits the rename itself; only the directory fsync is skipped.
// Unsupported by Chert: if anything else misbehaves, run the plain `chert` CLI on a Mac.
'use strict';
const fs = require('node:fs');
Object.defineProperty(process, 'platform', { value: 'linux', configurable: true });
if (typeof process.getuid !== 'function') process.getuid = () => 0;

const realLstat = fs.lstatSync;
fs.lstatSync = function patchedLstatSync(...args) {
  const s = realLstat.apply(this, args);
  if (s && typeof s.mode === 'number') {
    try { s.mode = s.mode & ~0o077; } catch { Object.defineProperty(s, 'mode', { value: s.mode & ~0o077 }); }
    if (typeof s.uid === 'number' && s.uid !== 0) { try { s.uid = 0; } catch { Object.defineProperty(s, 'uid', { value: 0 }); } }
  }
  return s;
};

const DIR_FDS = new Set(); let nextDirFd = 0x3fff0000;
const realOpen = fs.openSync, realFsync = fs.fsyncSync, realFdatasync = fs.fdatasyncSync, realClose = fs.closeSync, realFstat = fs.fstatSync;
function isDirFd(fd) {
  if (DIR_FDS.has(fd)) return true;
  try { return realFstat.call(fs, fd).isDirectory(); } catch { return false; }
}
fs.openSync = function patchedOpenSync(path, flags, mode) {
  try { return realOpen.call(this, path, flags, mode); }
  catch (e) {
    let isDir = false;
    try { isDir = fs.statSync(path).isDirectory(); } catch {}
    if (isDir && (e.code === 'EPERM' || e.code === 'EISDIR' || e.code === 'EACCES')) { const fd = nextDirFd++; DIR_FDS.add(fd); return fd; }
    throw e;
  }
};
fs.fsyncSync = function patchedFsyncSync(fd) {
  if (DIR_FDS.has(fd)) return;
  try { return realFsync.call(this, fd); }
  catch (e) { if ((e.code === 'EPERM' || e.code === 'EISDIR') && isDirFd(fd)) return; throw e; }
};
fs.fdatasyncSync = function patchedFdatasyncSync(fd) {
  if (DIR_FDS.has(fd)) return;
  try { return realFdatasync.call(this, fd); }
  catch (e) { if ((e.code === 'EPERM' || e.code === 'EISDIR') && isDirFd(fd)) return; throw e; }
};
fs.closeSync = function patchedCloseSync(fd) { if (DIR_FDS.has(fd)) { DIR_FDS.delete(fd); return; } return realClose.call(this, fd); };

require('node:module').syncBuiltinESMExports();   // the CLI is ESM: refresh the named-export bindings to the patched functions
