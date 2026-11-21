/* @ts-self-types="./cp_anywhere.d.ts" */

/**
 * A parsed pipeline that can process multiple image sets in one WASM instance.
 */
export class CppipePipeline {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CppipePipelineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_cppipepipeline_free(ptr, 0);
    }
    /**
     * @param {string} text
     */
    constructor(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.cppipepipeline_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        CppipePipelineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Inspect translated settings, including explicit save/export descriptors.
     * @returns {string}
     */
    plan() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.cppipepipeline_plan(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Process one field. This call does not modify the parsed pipeline.
     * @param {string} images_json
     * @param {Float64Array | number[]} pixels
     * @returns {string}
     */
    run(images_json, pixels) {
        let deferred4_0;
        let deferred4_1;
        try {
            let ptr0, len0, ptr1, len1;
            try {
                ptr0 = passStringToWasm0(images_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
                len0 = WASM_VECTOR_LEN;
                ptr1 = passArrayF64ToWasm0(pixels, wasm.__wbindgen_malloc);
                len1 = WASM_VECTOR_LEN;
            } catch (e) {
                if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
                if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 1, 1); }
                throw e;
            }
            const ret = wasm.cppipepipeline_run(this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var ptr3 = ret[0];
            var len3 = ret[1];
            if (ret[3]) {
                ptr3 = 0; len3 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred4_0 = ptr3;
            deferred4_1 = len3;
            return getStringFromWasm0(ptr3, len3);
        } finally {
            wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
        }
    }
}
if (Symbol.dispose) CppipePipeline.prototype[Symbol.dispose] = CppipePipeline.prototype.free;

/**
 * Result of `identifyPrimaryObjects`. Exposed as typed-array/scalar
 * getters, not a JSON string: the label arrays are the same size as
 * the input image (unlike `measure()`'s much smaller per-object
 * feature output), so JSON-encoding them would be wasteful.
 */
export class SegmentationResult {
    static __wrap(ptr) {
        const obj = Object.create(SegmentationResult.prototype);
        obj.__wbg_ptr = ptr;
        SegmentationResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SegmentationResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_segmentationresult_free(ptr, 0);
    }
    /**
     * @returns {number | undefined}
     */
    get guideThreshold() {
        const ret = wasm.segmentationresult_guideThreshold(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {Uint32Array}
     */
    get labels() {
        const ret = wasm.segmentationresult_labels(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get objectCount() {
        const ret = wasm.segmentationresult_objectCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Uint32Array}
     */
    get smallRemovedLabels() {
        const ret = wasm.segmentationresult_smallRemovedLabels(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Per-pixel corrected thresholds (Adaptive scope only); `null` for
     * Global scope, matching the Python binding's `threshold_map=None`.
     * @returns {Float64Array | undefined}
     */
    get thresholdMap() {
        const ret = wasm.segmentationresult_thresholdMap(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        }
        return v1;
    }
    /**
     * @returns {number}
     */
    get threshold() {
        const ret = wasm.segmentationresult_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Uint32Array}
     */
    get uneditedLabels() {
        const ret = wasm.segmentationresult_uneditedLabels(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) SegmentationResult.prototype[Symbol.dispose] = SegmentationResult.prototype.free;

/**
 * Apply a same-shaped 2D illumination function without epsilon substitution.
 * Comparisons deliberately preserve NaN and signed zero, unlike f64::max/min.
 * @param {Float64Array | number[]} pixels
 * @param {Float64Array | number[]} _function
 * @param {number} rows
 * @param {number} cols
 * @param {string} config_json
 * @returns {Float64Array}
 */
export function applyIllumination(pixels, _function, rows, cols, config_json) {
    const val4 = +rows;
    const val5 = +cols;
    let ptr0, len0, ptr1, len1, ptr2, len2;
    try {
        ptr0 = passArrayF64ToWasm0(pixels, wasm.__wbindgen_malloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = passArrayF64ToWasm0(_function, wasm.__wbindgen_malloc);
        len1 = WASM_VECTOR_LEN;
        ptr2 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        len2 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 1, 1); }
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 8, 8); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 8, 8); }
        throw e;
    }
    const ret = wasm.applyIllumination(ptr0, len0, ptr1, len1, val4, val5, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Calculate one 2D image's illumination function (CellProfiler "Each" mode).
 * Any nonzero mask byte is significant; absent mask preserves has_mask=False.
 * @param {Float64Array | number[]} pixels
 * @param {number} rows
 * @param {number} cols
 * @param {Uint8Array | number[] | null | undefined} mask
 * @param {string} config_json
 * @returns {Float64Array}
 */
export function calculateIllumination(pixels, rows, cols, mask, config_json) {
    const val4 = +rows;
    const val5 = +cols;
    let ptr0, len0, ptr1, len1, ptr2, len2;
    try {
        ptr0 = passArrayF64ToWasm0(pixels, wasm.__wbindgen_malloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = isLikeNone(mask) ? 0 : passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
        len1 = WASM_VECTOR_LEN;
        ptr2 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        len2 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 1, 1); }
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 1, 1); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 8, 8); }
        throw e;
    }
    const ret = wasm.calculateIllumination(ptr0, len0, val4, val5, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
}

/**
 * Translate original `.cppipe` text to an inspectable plan.
 * @param {string} text
 * @returns {string}
 */
export function compile_cppipe(text) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.compile_cppipe(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Segment typed-array pixel input using the real, oracle-verified
 * CellProfiler-compatible `identify_primary_objects` pipeline -- the
 * identical Rust core the Python binding (`cp_anywhere_python`) calls,
 * through the same `build_config`/`ConfigParams` settings surface (see
 * `cp_anywhere::segmentation::ConfigParams` for the full field list; any
 * field omitted from `config_json` uses CellProfiler's own equivalent
 * default, e.g. `threshold_method: "otsu"`). `pixels`/`mask` are plain
 * typed arrays (the bulk numeric data); `config_json` is a JSON object
 * carrying the many optional settings, matching `measureJson`'s own
 * "typed arrays for bulk data, JSON for small/sparse config" split.
 * @param {Float64Array | number[]} pixels
 * @param {number} rows
 * @param {number} cols
 * @param {Uint8Array | number[] | null | undefined} mask
 * @param {string} config_json
 * @returns {SegmentationResult}
 */
export function identifyPrimaryObjects(pixels, rows, cols, mask, config_json) {
    const val4 = +rows;
    const val5 = +cols;
    let ptr0, len0, ptr1, len1, ptr2, len2;
    try {
        ptr0 = passArrayF64ToWasm0(pixels, wasm.__wbindgen_malloc);
        len0 = WASM_VECTOR_LEN;
        ptr1 = isLikeNone(mask) ? 0 : passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
        len1 = WASM_VECTOR_LEN;
        ptr2 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        len2 = WASM_VECTOR_LEN;
    } catch (e) {
        if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 1, 1); }
        if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 1, 1); }
        if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 8, 8); }
        throw e;
    }
    const ret = wasm.identifyPrimaryObjects(ptr0, len0, val4, val5, ptr1, len1, ptr2, len2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return SegmentationResult.__wrap(ret[0]);
}

/**
 * @param {string} input
 * @returns {SegmentationResult}
 */
export function identifyPrimaryObjectsJson(input) {
    const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.identifyPrimaryObjectsJson(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return SegmentationResult.__wrap(ret[0]);
}

/**
 * Original eight-argument typed-array entry point. Keep existing callers
 * working; diagnostic neighbor layers and shape flags use measureWithOptions.
 * @param {Uint32Array | number[]} shape
 * @param {string[]} image_names
 * @param {Float64Array | number[]} images
 * @param {string[]} object_names
 * @param {Uint32Array | number[]} objects
 * @param {Uint8Array | number[] | null | undefined} mask
 * @param {string} image_id
 * @param {string[]} families
 * @returns {string}
 */
export function measure(shape, image_names, images, object_names, objects, mask, image_id, families) {
    let deferred10_0;
    let deferred10_1;
    try {
        let ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7;
        try {
            ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayJsValueToWasm0(image_names, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passArrayF64ToWasm0(images, wasm.__wbindgen_malloc);
            len2 = WASM_VECTOR_LEN;
            ptr3 = passArrayJsValueToWasm0(object_names, wasm.__wbindgen_malloc);
            len3 = WASM_VECTOR_LEN;
            ptr4 = passArray32ToWasm0(objects, wasm.__wbindgen_malloc);
            len4 = WASM_VECTOR_LEN;
            ptr5 = isLikeNone(mask) ? 0 : passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
            len5 = WASM_VECTOR_LEN;
            ptr6 = passStringToWasm0(image_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len6 = WASM_VECTOR_LEN;
            ptr7 = passArrayJsValueToWasm0(families, wasm.__wbindgen_malloc);
            len7 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr7) { wasm.__externref_drop_slice(ptr7, len7); wasm.__wbindgen_free(ptr7, len7 * 4, 4); }
            if (ptr6) { wasm.__wbindgen_free(ptr6, len6 * 1, 1); }
            if (ptr5) { wasm.__wbindgen_free(ptr5, len5 * 1, 1); }
            if (ptr4) { wasm.__wbindgen_free(ptr4, len4 * 4, 4); }
            if (ptr3) { wasm.__externref_drop_slice(ptr3, len3); wasm.__wbindgen_free(ptr3, len3 * 4, 4); }
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
            if (ptr1) { wasm.__externref_drop_slice(ptr1, len1); wasm.__wbindgen_free(ptr1, len1 * 4, 4); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.measure(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7);
        var ptr9 = ret[0];
        var len9 = ret[1];
        if (ret[3]) {
            ptr9 = 0; len9 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred10_0 = ptr9;
        deferred10_1 = len9;
        return getStringFromWasm0(ptr9, len9);
    } finally {
        wasm.__wbindgen_free(deferred10_0, deferred10_1, 1);
    }
}

/**
 * Run CellProfiler's 2D pairwise colocalization on JSON input -- the
 * same oracle-verified `cp_anywhere::colocalization` core the
 * Python binding (`cp_anywhere_python`) calls. Returns a JSON string
 * `{"image": {"<Correlation_... column name>": <float>, ...},
 * "objects": {"<column name>": [...one per object], ...}}`, matching
 * `cp_anywhere.measure_colocalization`'s own result shape.
 * @param {string} input
 * @returns {string}
 */
export function measureColocalizationJson(input) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.measureColocalizationJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Run CellProfiler's 2D granularity spectrum on typed-array/JSON input --
 * the same oracle-verified `cp_anywhere::granularity` core the
 * Python binding (`cp_anywhere_python`) calls. Returns a JSON string
 * `{"image": [...], "objects": {"<name>": [[...], ...]}}` (one inner
 * array per scale, ascending 1-based object order), matching
 * `cp_anywhere.measure_granularity`'s own result shape.
 * @param {string} input
 * @returns {string}
 */
export function measureGranularityJson(input) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.measureGranularityJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Run CellProfiler's 2D radial intensity distribution on typed-array/
 * JSON input -- the same oracle-verified
 * `cp_anywhere::intensity_distribution` core the Python binding
 * (`cp_anywhere_python`) calls. Returns a JSON string
 * `{"<RadialDistribution_... column name>": [...], ...}`, one value per
 * object in ascending 1-based object ID order, matching
 * `cp_anywhere.measure_intensity_distribution`'s own result shape.
 * @param {string} input
 * @returns {string}
 */
export function measureIntensityDistributionJson(input) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.measureIntensityDistributionJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * JSON-string overload kept for small inputs / debugging from plain JS
 * without constructing typed arrays by hand.
 * @param {string} input
 * @returns {string}
 */
export function measureJson(input) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.measureJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Run CellProfiler's 2D Haralick texture measurements on typed-array/
 * JSON input -- the same oracle-verified `cp_anywhere::texture`
 * core the Python binding (`cp_anywhere_python`) calls. Returns a JSON
 * string `{"image": {"<Texture_... column name>": <float>, ...},
 * "objects": {"<column name>": [...one per object], ...}}`, matching
 * `cp_anywhere.measure_texture`'s own result shape.
 * Set `float32: true` when `pixels` contains samples from a float32 plane.
 * @param {string} input
 * @returns {string}
 */
export function measureTextureJson(input) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(input, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.measureTextureJson(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Measure typed-array inputs and return the reference-compatible JSON
 * structure as a string. `shape`, `images`, `objects`, `mask`, and
 * `families` arrive as real typed arrays / plain JS values through
 * wasm-bindgen; only the (small) output is JSON-encoded, matching the
 * Python `MeasurementResult.to_json()` shape byte-for-byte in meaning.
 * @param {Uint32Array | number[]} shape
 * @param {string[]} image_names
 * @param {Float64Array | number[]} images
 * @param {string[]} object_names
 * @param {Uint32Array | number[]} objects
 * @param {Uint8Array | number[] | null | undefined} mask
 * @param {string} image_id
 * @param {string[]} families
 * @param {string[]} neighbors_scopes
 * @param {string[]} neighbors_expand
 * @param {string[]} full_object_names
 * @param {Uint32Array | number[]} full_objects
 * @param {boolean | null} [size_shape_calculate_zernikes]
 * @param {boolean | null} [size_shape_calculate_advanced]
 * @param {string | null} [neighbors_distances_json]
 * @param {string[] | null} [float32_images]
 * @returns {string}
 */
export function measureWithOptions(shape, image_names, images, object_names, objects, mask, image_id, families, neighbors_scopes, neighbors_expand, full_object_names, full_objects, size_shape_calculate_zernikes, size_shape_calculate_advanced, neighbors_distances_json, float32_images) {
    let deferred16_0;
    let deferred16_1;
    try {
        let ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, ptr8, len8, ptr9, len9, ptr10, len10, ptr11, len11, ptr12, len12, ptr13, len13;
        try {
            ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passArrayJsValueToWasm0(image_names, wasm.__wbindgen_malloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passArrayF64ToWasm0(images, wasm.__wbindgen_malloc);
            len2 = WASM_VECTOR_LEN;
            ptr3 = passArrayJsValueToWasm0(object_names, wasm.__wbindgen_malloc);
            len3 = WASM_VECTOR_LEN;
            ptr4 = passArray32ToWasm0(objects, wasm.__wbindgen_malloc);
            len4 = WASM_VECTOR_LEN;
            ptr5 = isLikeNone(mask) ? 0 : passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
            len5 = WASM_VECTOR_LEN;
            ptr6 = passStringToWasm0(image_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len6 = WASM_VECTOR_LEN;
            ptr7 = passArrayJsValueToWasm0(families, wasm.__wbindgen_malloc);
            len7 = WASM_VECTOR_LEN;
            ptr8 = passArrayJsValueToWasm0(neighbors_scopes, wasm.__wbindgen_malloc);
            len8 = WASM_VECTOR_LEN;
            ptr9 = passArrayJsValueToWasm0(neighbors_expand, wasm.__wbindgen_malloc);
            len9 = WASM_VECTOR_LEN;
            ptr10 = passArrayJsValueToWasm0(full_object_names, wasm.__wbindgen_malloc);
            len10 = WASM_VECTOR_LEN;
            ptr11 = passArray32ToWasm0(full_objects, wasm.__wbindgen_malloc);
            len11 = WASM_VECTOR_LEN;
            ptr12 = isLikeNone(neighbors_distances_json) ? 0 : passStringToWasm0(neighbors_distances_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len12 = WASM_VECTOR_LEN;
            ptr13 = isLikeNone(float32_images) ? 0 : passArrayJsValueToWasm0(float32_images, wasm.__wbindgen_malloc);
            len13 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr13) { wasm.__externref_drop_slice(ptr13, len13); wasm.__wbindgen_free(ptr13, len13 * 4, 4); }
            if (ptr12) { wasm.__wbindgen_free(ptr12, len12 * 1, 1); }
            if (ptr11) { wasm.__wbindgen_free(ptr11, len11 * 4, 4); }
            if (ptr10) { wasm.__externref_drop_slice(ptr10, len10); wasm.__wbindgen_free(ptr10, len10 * 4, 4); }
            if (ptr9) { wasm.__externref_drop_slice(ptr9, len9); wasm.__wbindgen_free(ptr9, len9 * 4, 4); }
            if (ptr8) { wasm.__externref_drop_slice(ptr8, len8); wasm.__wbindgen_free(ptr8, len8 * 4, 4); }
            if (ptr7) { wasm.__externref_drop_slice(ptr7, len7); wasm.__wbindgen_free(ptr7, len7 * 4, 4); }
            if (ptr6) { wasm.__wbindgen_free(ptr6, len6 * 1, 1); }
            if (ptr5) { wasm.__wbindgen_free(ptr5, len5 * 1, 1); }
            if (ptr4) { wasm.__wbindgen_free(ptr4, len4 * 4, 4); }
            if (ptr3) { wasm.__externref_drop_slice(ptr3, len3); wasm.__wbindgen_free(ptr3, len3 * 4, 4); }
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
            if (ptr1) { wasm.__externref_drop_slice(ptr1, len1); wasm.__wbindgen_free(ptr1, len1 * 4, 4); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 4, 4); }
            throw e;
        }
        const ret = wasm.measureWithOptions(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, ptr8, len8, ptr9, len9, ptr10, len10, ptr11, len11, isLikeNone(size_shape_calculate_zernikes) ? 0xFFFFFF : size_shape_calculate_zernikes ? 1 : 0, isLikeNone(size_shape_calculate_advanced) ? 0xFFFFFF : size_shape_calculate_advanced ? 1 : 0, ptr12, len12, ptr13, len13);
        var ptr15 = ret[0];
        var len15 = ret[1];
        if (ret[3]) {
            ptr15 = 0; len15 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred16_0 = ptr15;
        deferred16_1 = len15;
        return getStringFromWasm0(ptr15, len15);
    } finally {
        wasm.__wbindgen_free(deferred16_0, deferred16_1, 1);
    }
}

/**
 * Run original `.cppipe` text over named, decoded row-major planes.
 * `images_json` describes the name, shape, dtype, and optional role of each plane.
 * Use `image` (default), `objects` (retained), or `object_diagnostics` (small-removed).
 * Only images accept scaling. `pixels` concatenates the planes in descriptor order.
 * @param {string} text
 * @param {string} images_json
 * @param {Float64Array | number[]} pixels
 * @returns {string}
 */
export function run_cppipe(text, images_json, pixels) {
    let deferred5_0;
    let deferred5_1;
    try {
        let ptr0, len0, ptr1, len1, ptr2, len2;
        try {
            ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len0 = WASM_VECTOR_LEN;
            ptr1 = passStringToWasm0(images_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            len1 = WASM_VECTOR_LEN;
            ptr2 = passArrayF64ToWasm0(pixels, wasm.__wbindgen_malloc);
            len2 = WASM_VECTOR_LEN;
        } catch (e) {
            if (ptr2) { wasm.__wbindgen_free(ptr2, len2 * 8, 8); }
            if (ptr1) { wasm.__wbindgen_free(ptr1, len1 * 1, 1); }
            if (ptr0) { wasm.__wbindgen_free(ptr0, len0 * 1, 1); }
            throw e;
        }
        const ret = wasm.run_cppipe(ptr0, len0, ptr1, len1, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_string_get_7ce8eee7fd327545: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 1, len1, true);
            getDataViewMemory0().setInt32((arg0 >>> 0) + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./cp_anywhere_bg.js": import0,
    };
}

const CppipePipelineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_cppipepipeline_free(ptr, 1));
const SegmentationResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_segmentationresult_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('cp_anywhere_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
