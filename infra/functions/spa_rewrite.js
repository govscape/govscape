// CloudFront viewer-request function for the frontend bucket. The SvelteKit
// static build has no files for client-side routes such as /search or
// /preview/<id>, so any path whose last segment has no file extension is
// served the SPA entry point. Only attached to the default (non-/api) behavior.
function handler(event) {
  var request = event.request;
  var lastSegment = request.uri.split('/').pop();
  if (lastSegment.indexOf('.') === -1) {
    request.uri = '/index.html';
  }
  return request;
}
