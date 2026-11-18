"""Headless screenshots of Folium maps / the UI for the PDF report.

CDN assets are served from local npm copies when the machine has no CDN access
(set VENDOR_DIR); basemap tiles that cannot be fetched simply stay blank.
Usage: python report/screenshot.py <html_or_url> <out.png> [width height wait_ms]
"""
import os, sys
from playwright.sync_api import sync_playwright

V = os.environ.get("VENDOR_DIR", "/tmp/vend")
LOCAL = {
    "leaflet.js": f"{V}/leaflet-1.9.4/package/dist/leaflet.js",
    "leaflet.css": f"{V}/leaflet-1.9.4/package/dist/leaflet.css",
    "jquery-3.7.1.min.js": f"{V}/jquery-3.7.1/package/dist/jquery.min.js",
    "bootstrap.bundle.min.js": f"{V}/bootstrap-5.2.2/package/dist/js/bootstrap.bundle.min.js",
    "bootstrap.min.css": f"{V}/bootstrap-5.2.2/package/dist/css/bootstrap.min.css",
    "leaflet.awesome-markers.js": f"{V}/leaflet.awesome-markers-2.0.5/package/dist/leaflet.awesome-markers.js",
    "leaflet.awesome-markers.css": f"{V}/leaflet.awesome-markers-2.0.5/package/dist/leaflet.awesome-markers.css",
}


def shot(target, out, w=1400, h=900, wait=2500, actions=None):
    url = target if target.startswith("http") else "file://" + os.path.abspath(target)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROME", "/usr/bin/google-chrome"),
                              args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": w, "height": h})

        def route(r):
            u = r.request.url
            name = u.split("?")[0].rsplit("/", 1)[-1]
            if u.startswith(("file:", "http://localhost", "http://127.0.0.1")):
                return r.continue_()
            if name in LOCAL and os.path.exists(LOCAL[name]):
                ct = "text/css" if name.endswith(".css") else "application/javascript"
                return r.fulfill(path=LOCAL[name], content_type=ct)
            if os.environ.get("ONLINE") == "1":
                return r.continue_()
            return r.abort()
        pg.route("**/*", route)
        pg.goto(url)
        if actions:
            actions(pg)
        pg.wait_for_timeout(wait)
        pg.screenshot(path=out)
        b.close()


if __name__ == "__main__":
    a = sys.argv
    shot(a[1], a[2], *(int(x) for x in a[3:6]))
