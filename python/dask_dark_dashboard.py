"""Runtime dark theme for the Dask scheduler dashboard.

Applies Bokeh's native ``dark_minimal`` theme to every dashboard Document and
dark CSS to the page chrome, by swapping Dask's internal BokehApplication
factory. Call :func:`enable_dark_dashboard` BEFORE constructing any cluster
(the factory is read when the scheduler builds its dashboard).

Verified against distributed==2025.3.1 / bokeh==3.9.0 (the casa image).
The factory/template imports are not a stable public API, so everything is
wrapped defensively: on any failure this prints a warning and leaves the
stock light dashboard in place. Recipe:
ai-wiki/wiki/topics/dask_dashboard_dark_theme.md
"""

import functools

_DARK_CSS = """
<style>
  html, body, .content { background: #111827 !important; color: #e5e7eb !important; }
  .navbar, .navbar ul, .dropdown-content, .dropdown-content ul {
    background: #0b1220 !important;
  }
  .navbar { border-bottom: 1px solid #334155; }
  .navbar a, .navbar li a { color: #cbd5e1 !important; }
  .navbar li.active, .navbar li:hover, .navbar li.active a, .navbar li a:hover {
    background: #1e293b !important;
    color: #f8fafc !important;
  }
  #navbar-toggle-icon img { filter: brightness(0) invert(1); }
</style>
"""

_enabled = False


def enable_dark_dashboard(quiet=False):
    """Swap in the dark dashboard factory. Returns True if active."""
    global _enabled
    if _enabled:
        return True
    try:
        import dask
        from bokeh.application import Application
        from bokeh.application.handlers.function import FunctionHandler
        from bokeh.themes import built_in_themes
        try:
            from bokeh.server.util import create_hosts_allowlist
        except ImportError:  # older bokeh spelling
            from bokeh.server.util import (
                create_hosts_whitelist as create_hosts_allowlist,
            )
        import distributed.dashboard.core as dashboard_core
        import distributed.dashboard.scheduler as scheduler_dashboard
        from distributed.dashboard.components.scheduler import env as templates

        dark_status = templates.from_string(
            "{% extends 'status.html' %}"
            "{% block extra_resources %}{{ super() }}" + _DARK_CSS + "{% endblock %}"
        )
        dark_simple = templates.from_string(
            "{% extends 'simple.html' %}"
            "{% block extra_resources %}{{ super() }}" + _DARK_CSS + "{% endblock %}"
        )

        def dark_bokeh_application(applications, server, prefix="/", template_variables=None):
            template_variables = template_variables or {}
            prefix = "/" + prefix.strip("/") + "/" if prefix else "/"
            extra = {"prefix": prefix, **template_variables}

            def dark_view(document, view):
                # Dask's page constructors set their own light theme at the
                # end of construction, so the dark theme must go on afterward.
                view(server, extra, document)
                document.theme = built_in_themes["dark_minimal"]
                document.template = (
                    dark_status if document.title == "Dask: Status" else dark_simple
                )

            apps = {
                name: Application(FunctionHandler(functools.partial(dark_view, view=view)))
                for name, view in applications.items()
            }

            kwargs = dask.config.get(
                "distributed.scheduler.dashboard.bokeh-application"
            ).copy()
            allowed_origins = create_hosts_allowlist(
                kwargs.pop("allow_websocket_origin"), server.http_server.port
            )
            return dashboard_core.DaskBokehTornado(
                apps,
                prefix=prefix,
                use_index=False,
                extra_websocket_origins=allowed_origins,
                absolute_url="",
                **kwargs,
            )

        # scheduler_dashboard imported the factory by name -> replace both refs
        dashboard_core.BokehApplication = dark_bokeh_application
        scheduler_dashboard.BokehApplication = dark_bokeh_application
        _enabled = True
        if not quiet:
            print("[dask] dark dashboard theme enabled")
        return True
    except Exception as exc:
        if not quiet:
            print(f"[dask] dark dashboard unavailable ({type(exc).__name__}: {exc}) - using stock theme")
        return False
