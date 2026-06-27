import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { scoreWithPredictor } from "../src/staticApi.js";


test("browser LightGBM predictions match the Python artifact", async () => {
  const source = await readFile(
    new URL("../public/data/predictor.json", import.meta.url),
    "utf8",
  );
  const predictor = JSON.parse(source);

  for (const sample of predictor.verification) {
    const result = scoreWithPredictor(predictor, sample.customer);
    assert.ok(
      Math.abs(result.probability - sample.probability) < 1e-12,
      `Expected ${sample.probability}, received ${result.probability}`,
    );
  }
});
