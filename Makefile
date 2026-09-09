.PHONY: check workflow-check format smoke fetch-wands prepare-wands wands lexical-index index-wands \
	preflight-models embeddings vector-index query-onnx query-latency-diagnostic bm25 dense \
	int8-dense hybrid rrf significance fetch-sparse-model sparse-model sparse-precompute sparse-index sparse ml-commons-package \
	ml-commons-package-3.8 ml-commons-package-2.19 ml-commons-smoke ml-commons-smoke-3.8 \
	ml-commons-smoke-2.19 encoder-check encoder-production-lock up up-2.19 down \
	reproduce-wands reproduce-trec down-2.19 down-all

.PHONY: fetch-trec-product-search prepare-trec-product-search index-trec-product-search trec-bm25 fetch-trec-sparse-model trec-sparse-preflight trec-sparse-precompute trec-sparse-validate trec-sparse-query-model trec-sparse-smoke trec-sparse-index trec-sparse prepare-sku-slice index-sku-slice sku-slice routed-sparse-latency robustness prepare-reranker reranker index-three-way three-way ann-recall alternatives-significance facet-semantics
.PHONY: module-vector-ingestion-smoke module-vector-ingestion-decision radial-calibration-capture radial-judgments-prepare radial-judgments-bind radial-calibration-evaluate module-package module-package-verify

UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
MODULE_VECTOR_BENCHMARK_URL ?= http://127.0.0.1:9201
MODULE_VECTOR_BENCHMARK_IMAGE ?= opensearchproject/opensearch:3.8.0@sha256:bcc1797519726ceb6d651d4a3e60b7c30da91793914a8dfe75fd441d4f641509
MODULE_VECTOR_BENCHMARK_BATCH_SIZE ?= 64
MODULE_VECTOR_BENCHMARK_SMOKE_OUTPUT ?= results/module-opensearch-hybrid/base64-ingestion-smoke.json
MODULE_VECTOR_BENCHMARK_DECISION_OUTPUT ?= results/module-opensearch-hybrid/base64-ingestion-decision.json
RADIAL_CALIBRATION_CAPTURE ?= results/module-opensearch-hybrid/radial-calibration-capture.json
RADIAL_CALIBRATION_JUDGMENT_TEMPLATE ?= results/module-opensearch-hybrid/radial-calibration-judgment-template.json
RADIAL_CALIBRATION_JUDGMENTS ?= config/radial-calibration.judgments.json
RADIAL_CALIBRATION_INPUT ?= results/module-opensearch-hybrid/radial-calibration-bound.json
RADIAL_CALIBRATION_OUTPUT ?= results/module-opensearch-hybrid/radial-calibration-evaluation.json
RADIAL_CALIBRATION_SPECIFICATION ?= config/radial-calibration.example.json
RADIAL_CALIBRATION_URL ?= http://127.0.0.1:9201
MODULE_PACKAGE_ARCHIVE ?= dist/mage-os-module-opensearch-hybrid.zip
MODULE_PACKAGE_MANIFEST ?= dist/mage-os-module-opensearch-hybrid.manifest.json

check: encoder-check
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation ruff check .
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation mypy
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation pytest

workflow-check:
	go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12

encoder-check:
	PYTHONPATH=services/opensearch-hybrid-encoder/src UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models pytest services/opensearch-hybrid-encoder/tests
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen ruff check services/opensearch-hybrid-encoder/src services/opensearch-hybrid-encoder/tests services/opensearch-hybrid-encoder/deploy/preflight.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen mypy --config-file services/opensearch-hybrid-encoder/pyproject.toml services/opensearch-hybrid-encoder/src services/opensearch-hybrid-encoder/tests services/opensearch-hybrid-encoder/deploy/preflight.py

encoder-production-lock:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip compile services/opensearch-hybrid-encoder/pyproject.toml --extra production --python-version 3.13.15 --python-platform x86_64-manylinux_2_28 --only-binary :all: --generate-hashes --custom-compile-command 'make encoder-production-lock' --output-file services/opensearch-hybrid-encoder/requirements-production.lock

module-vector-ingestion-smoke:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/47_benchmark_module_vector_ingestion.py --url $(MODULE_VECTOR_BENCHMARK_URL) --output $(MODULE_VECTOR_BENCHMARK_SMOKE_OUTPUT) --documents 10000 --batch-sizes 16 32 64 --trials 1 --warmup-documents 1000 --opensearch-image $(MODULE_VECTOR_BENCHMARK_IMAGE) --topology same-host

module-vector-ingestion-decision:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/47_benchmark_module_vector_ingestion.py --url $(MODULE_VECTOR_BENCHMARK_URL) --output $(MODULE_VECTOR_BENCHMARK_DECISION_OUTPUT) --documents 50000 --batch-sizes $(MODULE_VECTOR_BENCHMARK_BATCH_SIZE) --trials 5 --warmup-documents 1000 --decision-run --opensearch-image $(MODULE_VECTOR_BENCHMARK_IMAGE) --topology same-host

radial-calibration-capture:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/49_capture_radial_similarity_calibration.py --url $(RADIAL_CALIBRATION_URL) --specification $(RADIAL_CALIBRATION_SPECIFICATION) --output $(RADIAL_CALIBRATION_CAPTURE)

radial-judgments-prepare:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/51_prepare_radial_similarity_judgments.py prepare --capture $(RADIAL_CALIBRATION_CAPTURE) --output $(RADIAL_CALIBRATION_JUDGMENT_TEMPLATE)

radial-judgments-bind:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/51_prepare_radial_similarity_judgments.py bind --capture $(RADIAL_CALIBRATION_CAPTURE) --judgments $(RADIAL_CALIBRATION_JUDGMENTS) --output $(RADIAL_CALIBRATION_INPUT)

radial-calibration-evaluate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/48_evaluate_radial_similarity_calibration.py --input $(RADIAL_CALIBRATION_INPUT) --output $(RADIAL_CALIBRATION_OUTPUT)

module-package:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/50_build_module_package.py build --module-root . --archive $(MODULE_PACKAGE_ARCHIVE) --manifest $(MODULE_PACKAGE_MANIFEST)

module-package-verify:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/50_build_module_package.py verify --module-root . --archive $(MODULE_PACKAGE_ARCHIVE) --manifest $(MODULE_PACKAGE_MANIFEST)

format:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen ruff format .
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen ruff check --fix .

up:
	./scripts/00_up.sh

up-2.19:
	docker compose -p opensearch-hybrid-os219 -f compose.opensearch-2.19.yml up -d --wait opensearch
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/00_verify_benchmark_profile.py --profile config/benchmark-2.19.toml --project opensearch-hybrid-os219 --expected-version 2.19.6 --url http://127.0.0.1:9219 --output results/environment/benchmark-profile-opensearch-2.19.json

smoke:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/01_smoke.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/01_verify.py

fetch-wands:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/02_fetch_wands.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/02_verify_wands.py

prepare-wands:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/03_prepare_wands.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/03_verify_wands.py

wands: fetch-wands prepare-wands

lexical-index:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/04_index_wands.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/04_verify_wands.py

preflight-models:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/09_preflight_model.py --model gte_modernbert_base
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/09_preflight_model.py --model granite_embedding_english_r2
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/09_preflight_model.py --model arctic_embed_m_v2

embeddings:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/07_embed_wands.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/08_verify_embeddings.py

vector-index:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/10_index_wands_vectors.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/10_verify_wands_vectors.py

index-wands: vector-index

query-onnx:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/13_build_query_onnx.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/13_verify_query_onnx.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/14_verify_query_parity.py --record-only
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/15_verify_self_retrieval.py

query-latency-diagnostic:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/16_measure_query_encoding.py

bm25:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/05_run_bm25.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/05_verify_bm25.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/06_crosscheck_bm25.py

dense:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/11_run_dense.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/12_verify_dense.py

int8-dense:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/16_run_int8_dense.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/16_assess_query_runtime.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/16_run_int8_dense.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/17_verify_int8_dense.py

hybrid:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/18_run_hybrid.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/19_verify_hybrid.py

rrf:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/21_run_rrf.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/22_verify_rrf.py

significance:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/20_run_significance.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/20_verify_significance.py

fetch-sparse-model:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/23_fetch_sparse_model.py

sparse-model:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/23_prepare_sparse_model.py

sparse-precompute: fetch-sparse-model
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/23_precompute_sparse.py

sparse-index:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/24_index_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/24_verify_sparse.py

sparse:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/25_run_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/25_verify_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/25_run_sparse_significance.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/25_verify_sparse_significance.py

ml-commons-package: ml-commons-package-3.8

ml-commons-package-3.8:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/26_package_ml_commons_dense.py --opensearch-version 3.8.0

ml-commons-package-2.19:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/26_package_ml_commons_dense.py --opensearch-version 2.19.6

ml-commons-smoke: ml-commons-smoke-3.8

ml-commons-smoke-3.8: ml-commons-package-3.8
	docker compose -f docker-compose.yml -f compose.bench.yml --profile ml-commons up -d --wait opensearch ml-commons-package-server
	OPENSEARCH_HYBRID_OS_URL=http://127.0.0.1:9201 UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/27_smoke_ml_commons_dense.py --opensearch-version 3.8.0 --package-manifest results/wands/ml-commons/arctic-package.json

ml-commons-smoke-2.19: ml-commons-package-2.19
	docker compose -p opensearch-hybrid-os219 -f compose.opensearch-2.19.yml --profile ml-commons up -d --wait opensearch ml-commons-package-server
	OPENSEARCH_HYBRID_OS_URL=http://127.0.0.1:9219 UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/27_smoke_ml_commons_dense.py --opensearch-version 2.19.6 --package-manifest results/wands/ml-commons/arctic-package-opensearch-2.x.json

fetch-trec-product-search:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/28_fetch_trec_product_search.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/28_verify_trec_product_search.py

prepare-trec-product-search:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/29_prepare_trec_product_search.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/29_verify_trec_product_search.py

index-trec-product-search:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/30_index_trec_product_search.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/30_verify_trec_product_search.py

trec-bm25:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra evaluation python scripts/31_run_trec_bm25.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra evaluation python scripts/31_verify_trec_bm25.py

fetch-trec-sparse-model:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/32_fetch_trec_sparse_model.py

trec-sparse-preflight: fetch-trec-sparse-model
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/32_precompute_trec_sparse.py --preflight-documents 1000

trec-sparse-precompute: fetch-trec-sparse-model
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/32_precompute_trec_sparse.py

trec-sparse-validate: fetch-trec-sparse-model
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/32_validate_trec_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/32_verify_trec_sparse_validation.py

trec-sparse-query-model:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/32_prepare_trec_sparse_query_model.py

trec-sparse-smoke:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/32_smoke_trec_sparse.py

trec-sparse-index:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/33_index_trec_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/33_verify_trec_sparse.py

trec-sparse:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/34_run_trec_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/34_verify_trec_sparse.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/34_run_trec_sparse_significance.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/34_verify_trec_sparse_significance.py

prepare-sku-slice:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/35_prepare_sku_slice.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen python scripts/35_verify_sku_slice.py

index-sku-slice:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/36_index_sku_slice.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/36_verify_sku_slice.py

sku-slice:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/37_run_sku_slice.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/37_verify_sku_slice.py

routed-sparse-latency:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/38_measure_routed_sparse_latency.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/38_verify_routed_sparse_latency.py

robustness:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/39_run_robustness.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/39_verify_robustness.py

prepare-reranker:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/40_prepare_reranker.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/40_verify_reranker_preparation.py

reranker:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/41_run_reranker.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/41_verify_reranker.py

index-three-way:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/42_index_three_way.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models python scripts/42_verify_three_way.py

three-way:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/43_run_three_way.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/43_verify_three_way.py

ann-recall:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/44_run_ann_recall.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/44_verify_ann_recall.py

alternatives-significance:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/45_run_alternatives_significance.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/45_verify_alternatives_significance.py

facet-semantics:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/46_run_facet_semantics.py
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen --extra models --extra evaluation python scripts/46_verify_facet_semantics.py

reproduce-wands:
	$(MAKE) up
	$(MAKE) wands
	$(MAKE) preflight-models
	$(MAKE) embeddings
	$(MAKE) vector-index
	$(MAKE) bm25
	$(MAKE) dense
	$(MAKE) query-onnx
	$(MAKE) int8-dense
	$(MAKE) hybrid
	$(MAKE) rrf
	$(MAKE) significance
	$(MAKE) fetch-sparse-model
	$(MAKE) sparse-precompute
	$(MAKE) sparse-model
	$(MAKE) sparse-index
	$(MAKE) sparse
	$(MAKE) prepare-sku-slice
	$(MAKE) index-sku-slice
	$(MAKE) sku-slice
	$(MAKE) robustness
	$(MAKE) index-three-way
	$(MAKE) three-way
	$(MAKE) prepare-reranker
	$(MAKE) reranker
	$(MAKE) alternatives-significance
	$(MAKE) ann-recall
	$(MAKE) facet-semantics
	$(MAKE) routed-sparse-latency

reproduce-trec:
	$(MAKE) up
	$(MAKE) up-2.19
	$(MAKE) fetch-trec-product-search
	$(MAKE) prepare-trec-product-search
	$(MAKE) index-trec-product-search
	$(MAKE) trec-bm25
	$(MAKE) fetch-trec-sparse-model
	$(MAKE) trec-sparse-preflight
	$(MAKE) trec-sparse-query-model
	$(MAKE) trec-sparse-smoke
	$(MAKE) trec-sparse-precompute
	$(MAKE) trec-sparse-validate
	$(MAKE) trec-sparse-index
	$(MAKE) trec-sparse

down:
	docker compose -f docker-compose.yml -f compose.bench.yml down

down-2.19:
	docker compose -p opensearch-hybrid-os219 -f compose.opensearch-2.19.yml down

down-all:
	$(MAKE) down
	$(MAKE) down-2.19
