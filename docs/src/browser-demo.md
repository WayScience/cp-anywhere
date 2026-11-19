# Cell profiling in your browser

Run illumination correction, segmentation, and measurements on a real microscopy crop. cp-anywhere runs in WebAssembly on this page. No Python kernel, server-side analysis, or image uploads are needed.

Sample: [BBBC007v1](https://bbbc.broadinstitute.org/BBBC007), Drosophila Kc167 cells, field `A9 p10`, 128 × 128 pixels. DNA marks the nuclei. Actin shows the cell bodies.

The demo adds a **simulated lighting gradient** to both channels. It estimates a correction from a separate simulated flat field, not from the cell images. Compare the shaded and corrected images with the original on the same intensity scale. This shows recovery from known shading, not calibration of a real microscope.

```{raw} html
<section id="wasm-demo" aria-label="Interactive cell profiling">
  <p role="status">Loading the WebAssembly demo…</p>
  <noscript>Enable JavaScript to run the WebAssembly demo.</noscript>
  <div data-results hidden>
    <h2>1. Compare illumination</h2>
    <p><label>Channel for comparison
      <select name="channel"><option>DNA</option><option>Actin</option></select>
    </label></p>
    <div class="wasm-demo-grid" data-channel-grid>
      <figure><canvas width="128" height="128" role="img" aria-label="Simulated shaded image" data-image="shaded"></canvas><figcaption>Simulated shading</figcaption></figure>
      <figure><canvas width="128" height="128" role="img" aria-label="Image after illumination correction" data-image="corrected"></canvas><figcaption>After correction</figcaption></figure>
      <figure><canvas width="128" height="128" role="img" aria-label="Original reference image" data-image="original"></canvas><figcaption>Original reference</figcaption></figure>
    </div>
    <p>Mean absolute pixel error vs. original for <span data-channel>DNA</span> (0–1 scale):
      <strong data-error="shaded"></strong> before correction and <strong data-error="corrected"></strong> after.</p>
    <h2>2. Inspect segmentation</h2>
    <p><label><input type="checkbox" name="correction" checked> Use illumination correction</label>
    Clear this box to rerun the same pipeline on the shaded images.</p>
    <div class="wasm-demo-grid">
      <figure><canvas width="128" height="128" role="img" aria-label="Nuclei boundaries on DNA" data-overlay="Nuclei"></canvas><figcaption>Nuclei on DNA</figcaption></figure>
      <figure><canvas width="128" height="128" role="img" aria-label="Cell boundaries on actin" data-overlay="Cells"></canvas><figcaption>Cells on actin</figcaption></figure>
      <figure><canvas width="128" height="128" role="img" aria-label="Cytoplasm boundaries on actin" data-overlay="Cytoplasm"></canvas><figcaption>Cytoplasm on actin</figcaption></figure>
    </div>
    <h2>3. Review measurements</h2>
    <table class="table">
      <caption>Cytoplasm intensity for the current run</caption>
      <thead><tr><th scope="col">Object ID</th><th scope="col">Mean actin intensity (0–1)</th></tr></thead>
      <tbody></tbody>
    </table>
    <p><a data-csv download>Download measurements (CSV)</a> · <a href="_static/browser-demo/demo.cppipe" download>Download the example pipeline (.cppipe)</a></p>
  </div>
</section>
<script type="module" src="_static/browser-demo.js"></script>
```

The saved `.cppipe` identifies nuclei, expands them to cells using actin, subtracts nuclei from cells, and measures cytoplasm intensity. The page applies illumination correction before it passes the images to the pipeline. The `Stain` input in the pipeline is the actin channel. These settings are for this example, not recommended defaults for new experiments.

See the [Python notebook](cell-profiling-anywhere.ipynb) for the same microscopy crop, more measurements, and image and table exports.
