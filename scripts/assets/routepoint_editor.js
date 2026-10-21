(function () {
  "use strict";

  const GRAPH_ID = "route-graph";
  const DRAG_STORE = "drag-store";

  function getGraph() {
    const host = document.getElementById(GRAPH_ID);
    return host ? host.querySelector(".js-plotly-plot") : null;
  }

  function setDragStore(payload) {
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(DRAG_STORE, { data: payload });
      return;
    }
    console.warn("dash_clientside.set_props is unavailable");
  }

  function traceIndexForNode(gd, node) {
    const traceGroup = node.closest("g.trace");
    if (!traceGroup) {
      return -1;
    }
    const groups = Array.prototype.slice.call(
      gd.querySelectorAll(".scatterlayer > g.trace")
    );
    return groups.indexOf(traceGroup);
  }

  function pointIndexForNode(node) {
    const traceGroup = node.closest(".trace");
    if (!traceGroup) {
      return -1;
    }
    const points = Array.prototype.slice.call(
      traceGroup.querySelectorAll("path.point")
    );
    return points.indexOf(node);
  }

  function axisSpans(gd) {
    const layout = gd._fullLayout;
    if (!layout || !layout.xaxis || !layout.yaxis) {
      return null;
    }
    const ax = layout.xaxis;
    const ay = layout.yaxis;
    const xr = ax._rl || ax.range;
    const yr = ay._rl || ay.range;
    if (!xr || !yr || !ax._length || !ay._length) {
      return null;
    }
    const xSpan = Math.abs(Number(xr[1]) - Number(xr[0]));
    const ySpan = Math.abs(Number(yr[1]) - Number(yr[0]));
    if (!isFinite(xSpan) || !isFinite(ySpan) || xSpan <= 0 || ySpan <= 0) {
      return null;
    }
    return {
      xPerPx: xSpan / Number(ax._length),
      yPerPx: ySpan / Number(ay._length),
    };
  }

  function onMouseDown(event) {
    if (event.button !== 0) {
      return;
    }
    const gd = event.currentTarget;
    const target = event.target;
    if (!gd || !(target instanceof Element)) {
      return;
    }
    const pointNode = target.closest("path.point");
    if (!pointNode) {
      return;
    }
    const traceIndex = traceIndexForNode(gd, pointNode);
    if (traceIndex < 0 || !gd.data || !gd.data[traceIndex]) {
      return;
    }
    let trace = gd.data[traceIndex];
    let meta = trace.meta;
    let index = 0;
    if (meta === "route") {
      index = pointIndexForNode(pointNode);
    } else if (meta === "selected") {
      const custom = trace.customdata && trace.customdata[0];
      const pointKind = custom ? custom[0] : "route";
      if (pointKind === "route") {
        const routeIndex = gd.data.findIndex(function (item) {
          return item && item.meta === "route";
        });
        if (routeIndex < 0) {
          return;
        }
        trace = gd.data[routeIndex];
        meta = "route";
        index = custom ? parseInt(custom[1], 10) : 0;
      } else if (pointKind === "first" || pointKind === "last") {
        meta = pointKind;
        index = 0;
      } else {
        return;
      }
    } else if (trace.name === "flagged points") {
      const routeIndex = gd.data.findIndex(function (item) {
        return item && item.meta === "route";
      });
      const custom = trace.customdata && trace.customdata[0];
      if (routeIndex < 0 || !custom) {
        return;
      }
      trace = gd.data[routeIndex];
      meta = "route";
      index = parseInt(custom[0], 10);
    } else if (meta !== "first" && meta !== "last") {
      return;
    }
    if (index < 0 || !trace.x || !trace.y || index >= trace.x.length) {
      return;
    }
    const startX = Number(trace.x[index]);
    const startY = Number(trace.y[index]);
    if (!isFinite(startX) || !isFinite(startY)) {
      return;
    }
    const spans = axisSpans(gd);
    if (!spans) {
      return;
    }
    const startClientX = event.clientX;
    const startClientY = event.clientY;
    let moved = false;
    let lastDx = 0;
    let lastDy = 0;
    const transform = pointNode.getAttribute("transform") || "";
    const transformMatch = /translate\(\s*([-0-9.]+)[,\s]+([-0-9.]+)\s*\)/.exec(
      transform
    );
    const basePx = transformMatch
      ? [parseFloat(transformMatch[1]), parseFloat(transformMatch[2])]
      : null;
    const previousSelect = document.body.style.userSelect;
    const previousCursor = document.body.style.cursor;

    event.preventDefault();
    event.stopPropagation();
    if (event.stopImmediatePropagation) {
      event.stopImmediatePropagation();
    }
    document.body.style.userSelect = "none";
    document.body.style.cursor = "grabbing";

    function onMove(moveEvent) {
      lastDx = moveEvent.clientX - startClientX;
      lastDy = moveEvent.clientY - startClientY;
      if (Math.abs(lastDx) + Math.abs(lastDy) > 1) {
        moved = true;
      }
      if (basePx) {
        pointNode.setAttribute(
          "transform",
          "translate(" +
            (basePx[0] + lastDx).toFixed(2) +
            "," +
            (basePx[1] + lastDy).toFixed(2) +
            ")"
        );
      }
    }

    function onUp(upEvent) {
      document.removeEventListener("mousemove", onMove, true);
      document.removeEventListener("mouseup", onUp, true);
      document.body.style.userSelect = previousSelect;
      document.body.style.cursor = previousCursor;
      const dx = upEvent.clientX - startClientX;
      const dy = upEvent.clientY - startClientY;
      const newX = startX + dx * spans.xPerPx;
      const newY = startY - dy * spans.yPerPx;
      if (!isFinite(newX) || !isFinite(newY)) {
        return;
      }
      if (moved) {
        const swallowClick = function (clickEvent) {
          clickEvent.stopPropagation();
          clickEvent.preventDefault();
        };
        document.addEventListener("click", swallowClick, true);
        window.setTimeout(function () {
          document.removeEventListener("click", swallowClick, true);
        }, 0);
      }
      setDragStore({
        kind: meta === "route" ? "route" : meta,
        index: index,
        x: newX,
        y: newY,
        moved: moved,
        seq: Date.now(),
      });
    }

    document.addEventListener("mousemove", onMove, true);
    document.addEventListener("mouseup", onUp, true);
  }

  function attachDrag() {
    const gd = getGraph();
    if (!gd || gd.__routepointEditorDrag) {
      return;
    }
    gd.__routepointEditorDrag = true;
    gd.addEventListener("mousedown", onMouseDown, true);
  }

  function clickButton(id) {
    const button = document.getElementById(id);
    if (button) {
      button.click();
    }
  }

  window.addEventListener("beforeunload", function (event) {
    if (!document.getElementById(GRAPH_ID)) {
      return;
    }
    const indicator = document.getElementById("dirty-indicator");
    if (indicator && indicator.textContent.indexOf("Unsaved") !== -1) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  document.addEventListener("keydown", function (event) {
    if (!document.getElementById(GRAPH_ID)) {
      return;
    }
    if (!event.ctrlKey && !event.metaKey) {
      return;
    }
    const key = (event.key || "").toLowerCase();
    const tag = (event.target && event.target.tagName
      ? event.target.tagName
      : ""
    ).toLowerCase();
    const editing =
      tag === "input" ||
      tag === "textarea" ||
      (event.target && event.target.isContentEditable);
    if (editing && key !== "s") {
      return;
    }
    if (key === "s") {
      event.preventDefault();
      clickButton("save-btn");
      return;
    }
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      clickButton("undo-btn");
      return;
    }
    if (key === "y" || (key === "z" && event.shiftKey)) {
      event.preventDefault();
      clickButton("redo-btn");
    }
  });

  window.setInterval(attachDrag, 500);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachDrag);
  } else {
    attachDrag();
  }
})();
