from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from poc.module_package import collect_payload

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT
ENCODER = ROOT / "services/opensearch-hybrid-encoder"


def _xml(path: str) -> ET.Element:
    return ET.parse(MODULE / path).getroot()


def test_module_identity_is_canonical_and_self_contained() -> None:
    composer = json.loads((MODULE / "composer.json").read_text())
    registration = (MODULE / "registration.php").read_text()
    module_xml = _xml("etc/module.xml")
    module = module_xml.find("module")

    assert composer["name"] == "mage-os/module-opensearch-hybrid"
    assert composer["autoload"]["psr-4"] == {"MageOS\\OpenSearchHybrid\\": ""}
    assert "MageOS_OpenSearchHybrid" in registration
    assert module is not None
    assert module.attrib["name"] == "MageOS_OpenSearchHybrid"
    assert "../" not in json.dumps(composer)


def test_search_layer_engine_factories_extend_real_virtual_types() -> None:
    di = _xml("etc/di.xml")
    expected = {
        "elasticsearchLayerCategoryItemCollectionProvider": (
            "Magento\\Elasticsearch\\Model\\Layer\\Category\\ItemCollectionProvider"
        ),
        "elasticsearchLayerSearchItemCollectionProvider": (
            "Magento\\Elasticsearch\\Model\\Layer\\Search\\ItemCollectionProvider"
        ),
    }

    for name, base_type in expected.items():
        virtual_type = di.find(f"virtualType[@name='{name}']")
        assert virtual_type is not None
        assert virtual_type.attrib["type"] == base_type
        assert di.find(f"type[@name='{name}']") is None


def test_production_sources_have_no_legacy_identity() -> None:
    forbidden = ("MageOS_RocketSearch", "RocketSearch\\", "rocket_search")
    production_files = [
        path
        for path in (MODULE / entry.path for entry in collect_payload(MODULE))
        if path.is_file()
        and "Test" not in path.parts
        and "vendor" not in path.relative_to(MODULE).parts
        and path.name != "README.md"
    ]

    for path in production_files:
        content = path.read_text(errors="ignore")
        for value in forbidden:
            assert value not in content, f"{value} found in {path.relative_to(MODULE)}"


def test_persisted_database_identifiers_fit_mysql_limit() -> None:
    schema = _xml("etc/db_schema.xml")

    for table in schema.findall("table"):
        assert len(table.attrib["name"]) <= 64
        assert table.attrib["name"].startswith("mageos_opensearch_hybrid_")
        for element in table.findall(".//*[@name]"):
            assert len(element.attrib["name"]) <= 64
        for element in table.findall(".//*[@referenceId]"):
            assert len(element.attrib["referenceId"]) <= 64


def test_queue_topic_and_queue_names_match_across_configuration() -> None:
    communication = _xml("etc/communication.xml")
    topology = _xml("etc/queue_topology.xml")
    publishers = _xml("etc/queue_publisher.xml")
    consumers = _xml("etc/queue_consumer.xml")

    declared_topics = {topic.attrib["name"] for topic in communication.findall("topic")}
    published_topics = {publisher.attrib["topic"] for publisher in publishers.findall("publisher")}
    bound_topics = {binding.attrib["topic"] for binding in topology.findall(".//binding")}
    bound_queues = {
        binding.attrib["destination"]
        for binding in topology.findall(".//binding")
        if binding.attrib.get("destinationType") == "queue"
    }
    consumed_queues = {consumer.attrib["queue"] for consumer in consumers.findall("consumer")}

    assert published_topics <= declared_topics
    assert bound_topics == declared_topics
    assert consumed_queues <= bound_queues
    assert consumed_queues == {
        "mageos.opensearch_hybrid.embedding",
        "mageos.opensearch_hybrid.correctness_priority",
    }


def test_async_event_payload_resolves_job_identity_only() -> None:
    async_events = _xml("etc/async_events.xml")
    event = async_events.find("async_event")
    service = event.find("service") if event is not None else None

    assert event is not None
    assert event.attrib["name"] == "mageos.opensearch_hybrid.embedding.batch.v1"
    assert service is not None
    assert service.attrib == {
        "class": "MageOS\\OpenSearchHybrid\\Api\\JobPayloadInterface",
        "method": "get",
    }


def test_frozen_retrieval_contract_matches_registered_dense_recipe() -> None:
    contract = json.loads((MODULE / "etc/opensearch_hybrid_contract.json").read_text())

    assert contract["model"] == {
        "id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "revision": "95c2741480856aa9666782eb4afe11959938017f",
        "dimension": 256,
        "similarity": "cosine",
        "query_prefix": "query: ",
        "query_template": "{query}",
        "document_template": "{title}. {description}. {attributes}",
        "source_recipe_version": "mageos-v1",
        "lexical_brand_attributes": ["manufacturer"],
        "semantic_feature_attributes": ["color", "material"],
    }
    assert contract["fusion"] == {
        "query_type": "hybrid",
        "normalization": "min_max",
        "combination": "arithmetic_mean",
        "weights": [0.3, 0.7],
    }
    assert contract["result_contract"]["candidate_universe"] == "native_opensearch_top_100"
    assert contract["result_contract"]["totals_and_aggregations"] == "native_opensearch"
    assert contract["result_contract"]["lexical_score_source"] == "native_candidate_score"
    assert contract["result_contract"]["merchant_search_weight_control"] == "native_catalog_search"
    assert contract["result_contract"]["version"] == "native-candidate-score-rerank-v5"
    assert contract["result_contract"]["hybrid_request_names"] == [
        "quick_search_container",
        "graphql_product_search",
    ]
    assert contract["result_contract"]["request_sort_contracts"] == {
        "quick_search_container": [
            "native_relevance",
            "storefront_default_sort_alias",
        ],
        "graphql_product_search": ["native_relevance"],
    }
    assert contract["result_contract"]["storefront_default_sort_alias"] == (
        "is_out_of_stock_asc_position_asc"
    )
    assert contract["result_contract"]["availability_ordering"] == (
        "salable_first_stable_partition"
    )
    assert contract["result_contract"]["availability_ordering_scope"] == (
        "storefront_default_sort_alias_only"
    )
    assert contract["lexical"] == {
        "candidate_selection": "native_opensearch_top_100",
        "score_source": "native_candidate_score",
        "merchant_control": "catalog_search_attribute_weights",
        "module_field_boosts": None,
    }
    assert contract["mapping"]["properties"]["embedding"]["dimension"] == 256


def test_every_new_php_source_is_strict_and_not_final() -> None:
    for path in MODULE.rglob("*.php"):
        if "vendor" in path.relative_to(MODULE).parts:
            continue
        content = path.read_text()
        assert "declare(strict_types=1);" in content, path.relative_to(MODULE)
        assert "final class" not in content, path.relative_to(MODULE)


def test_query_encoding_uses_the_generation_bound_endpoint() -> None:
    encoder = (MODULE / "Model/Encoder/HttpQueryEncoder.php").read_text()

    assert "validateEncoderEndpoint((string)$generation['encoder_endpoint'])" in encoder
    assert "encoderEndpoint() . '/v1/embed/query'" not in encoder


def test_document_encoding_is_generation_bound_batched_and_requires_production_identity() -> None:
    interface = (MODULE / "Api/DocumentEncoderInterface.php").read_text()
    encoder = (MODULE / "Model/Encoder/HttpDocumentEncoder.php").read_text()
    processor = (MODULE / "Model/Queue/EmbeddingJobProcessor.php").read_text()
    di = _xml("etc/di.xml")

    preference = di.find(
        "preference[@for='MageOS\\OpenSearchHybrid\\Api\\DocumentEncoderInterface']"
    )
    assert preference is not None
    assert preference.attrib["type"] == (
        "MageOS\\OpenSearchHybrid\\Model\\Encoder\\HttpDocumentEncoder"
    )
    assert "public function encode(array $documents, array $generation): array" in interface
    assert "/v1/embed/documents" in encoder
    assert "production_eligible" in encoder
    assert "encoder_identity_digest" in encoder
    assert "count($vectors) !== count($documents)" in encoder
    assert "MAX_BATCH_SIZE = 64" in encoder
    assert "count($documents) > self::MAX_BATCH_SIZE" in encoder
    assert "array_chunk($documents, $this->config->batchSize())" in encoder
    assert "$this->documentEncoder->encode(" in processor
    assert "The qualified production document encoder is not implemented" not in processor

    config = (MODULE / "Model/Config.php").read_text()
    assert "boundedInt(self::XML_PATH_BATCH_SIZE, 1, 64)" in config
    defaults = _xml("etc/config.xml")
    assert defaults.findtext("default/mageos_opensearch_hybrid/encoder/batch_size") == "16"


def test_production_encoder_identity_is_operator_pinned_and_manifest_verified() -> None:
    verifier = (MODULE / "Model/Encoder/EncoderIdentityVerifier.php").read_text()
    client = (MODULE / "Model/Encoder/EncoderIdentityClient.php").read_text()
    command = (MODULE / "Console/EncoderIdentityCommand.php").read_text()
    di = _xml("etc/di.xml")
    registered = di.find(
        ".//item[@name='mageos_opensearch_hybrid_encoder_identity']"
    )

    assert "identity_manifest" in verifier
    assert "digestValue($manifest)" in verifier
    assert "linux/amd64" in verifier
    assert "REQUIRED_EVIDENCE_ROLES" in verifier
    assert "fetchAt($this->config->encoderEndpoint(), $expectedDigest)" in client
    assert "validateEncoderEndpoint($endpoint) . '/v1/identity'" in client
    assert "encoder:identity" in command
    assert "'expect'," in command
    assert "Provide the exact lowercase 64-character digest with --expect" in command
    assert "operator-pinned identity" in command
    assert registered is not None
    assert registered.text == "MageOS\\OpenSearchHybrid\\Console\\EncoderIdentityCommand"


def test_production_generation_creation_requires_a_verified_pinned_identity() -> None:
    build = (MODULE / "Model/Generation/BuildService.php").read_text()
    command = (MODULE / "Console/BuildCommand.php").read_text()
    runner = (MODULE / "dev/ci/run.sh").read_text()

    assert "buildProduction(int $storeId, string $expectedIdentityDigest)" in build
    production = build[build.index("public function buildProduction"):]
    assert production.index("identityClient->fetchAt") < production.index("createGeneration(")
    assert "'is_fake' => $isFake ? 1 : 0" in build
    assert "'encoder-identity'," in command
    assert "buildProduction" in command
    assert "assert-production-build-gate.php" in runner


def test_production_generation_build_has_a_read_only_impact_preview() -> None:
    build = (MODULE / "Model/Generation/BuildService.php").read_text()
    command = (MODULE / "Console/BuildCommand.php").read_text()

    signature = (
        "public function planProduction(int $storeId, string $expectedIdentityDigest): array"
    )
    assert signature in build
    start = build.index("public function planProduction")
    end = build.index("public function buildProduction")
    preview = build[start:end]
    assert "identityClient->fetchAt" in preview
    assert "indexManager->assertSupportedVersion" in preview
    assert "countEligible" in preview
    assert "expected_total_embedding_batches" in preview
    assert "initial_outbox_jobs_without_consumer_progress" in preview
    assert "raw_float32_bytes" in preview
    assert "base64_bytes" in preview
    assert "'activation'" in preview
    assert "'changes' => 0" in preview
    assert "'current_mode'" in preview
    assert "'current_generation_id'" in preview
    assert "'planned_mode'" in preview
    assert "'planned_generation_id'" in preview
    assert "insert(" not in preview
    assert "update(" not in preview
    assert "delete(" not in preview
    assert "'dry-run'," in command
    assert "planProduction" in command


def test_production_generation_build_requires_the_current_state_bound_preview() -> None:
    build = (MODULE / "Model/Generation/BuildService.php").read_text()
    command = (MODULE / "Console/BuildCommand.php").read_text()

    assert "'confirmation_token'" in build
    assert "'scope'" in build
    assert "'scope_digest' => $scopeDigest" in build
    assert "build-production-%d-%s" in build
    assert "buildProductionConfirmed(" in build
    confirmed = build[build.index("public function buildProductionConfirmed"):]
    assert confirmed.index("planProduction(") < confirmed.index("hash_equals(")
    assert (
        "The production build confirmation does not match the current exact preview." in confirmed
    )
    assert "'confirm'," in command
    assert "buildProductionConfirmed" in command
    assert "Use --dry-run for preview or --confirm with its exact token for apply." in command


def test_large_catalog_qualification_is_isolated_and_opt_in() -> None:
    script = (MODULE / "dev/ci/assert-large-catalog-build.php").read_text()
    graphql = (MODULE / "dev/ci/assert-large-catalog-graphql.php").read_text()
    runner = (MODULE / "dev/ci/run-large-catalog.sh").read_text()
    workflow = (ROOT / ".github/workflows/module-opensearch-hybrid.yml").read_text()

    assert "targetProducts = isset($argv[2]) ? (int)$argv[2] : 10_000" in script
    assert "no writable generation" in script
    assert "cataloginventory_stock_item" in script
    assert "coverage_complete" in script
    assert "coverage_failed" in script
    assert "resultContractDigest" in script
    assert 'search: "qualification"' in graphql
    assert "GraphQL price aggregations do not account for the result set" in graphql
    assert "assert-large-catalog-build.php" in runner
    assert "assert-large-catalog-graphql.php" in runner
    assert "run_large_catalog:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "inputs.run_large_catalog" in workflow
    assert "matrix.fixture == 'fresh-3.4'" in workflow
    assert "run-large-catalog.sh" in workflow


def test_registered_catalog_preparer_is_approval_gated_and_reversible() -> None:
    script = (
        MODULE / "dev/qualification/prepare-registered-catalog.php"
    ).read_text()

    assert "mageos-radial-registered-" in script
    assert "TARGET_PRODUCTS = 50_000" in script
    assert "MODE_PLAN" in script
    assert "MODE_APPLY" in script
    assert "MODE_ROLLBACK_PREVIEW" in script
    assert "MODE_ROLLBACK" in script
    assert "confirmation_token" in script
    assert "hash('sha256'" in script
    assert "GET_LOCK" in script
    assert "RELEASE_LOCK" in script
    assert "insertMultiple" in script
    assert "catalog_product_website" in script
    assert "inventory_source_item" in script
    assert "cataloginventory_stock_item" in script
    assert "catalog_product_entity_varchar" in script
    assert "catalog_product_entity_int" in script
    assert "catalog_product_entity_decimal" in script
    assert "catalog_product_entity_text" in script
    assert "catalogsearch_fulltext" in script
    assert "fixture_eligible_products" in script
    assert "The prefixed fixture is incomplete or inconsistent" in script
    assert "fixture_consistent" in script
    assert "fixture_inconsistencies" in script
    assert "if ($requireApplySafeState && !$fixtureConsistent)" in script
    assert "if ($requireApplySafeState && $eligibleProducts > TARGET_PRODUCTS)" in script


def test_inactive_full_build_refresh_suppression_is_durable_and_activation_safe() -> None:
    schema = _xml("etc/db_schema.xml")
    build = (MODULE / "Model/Generation/BuildService.php").read_text()
    refresh = (MODULE / "Model/Generation/BuildRefreshService.php").read_text()
    validation = (MODULE / "Model/Generation/ValidationService.php").read_text()
    index_manager = (MODULE / "Model/OpenSearch/IndexManager.php").read_text()
    activation = (MODULE / "Model/Activation/ActivationService.php").read_text()
    metrics = (MODULE / "Model/Operations/OperationalMetricsService.php").read_text()
    integration = (MODULE / "dev/ci/assert-build-resume.php").read_text()

    progress = schema.find(
        "table[@name='mageos_opensearch_hybrid_generation_progress']"
    )
    assert progress is not None
    for column in (
        "build_refresh_interval",
        "normal_refresh_interval",
        "refresh_state",
    ):
        assert progress.find(f"column[@name='{column}']") is not None
    assert "buildRefreshService->suppress" in build
    assert "seeding_state" in refresh
    assert "SUPPRESSED" in refresh
    assert "RESTORED" in refresh
    assert "RESTORE_FAILED" in refresh
    assert validation.index("buildRefreshService->restore") < validation.index(
        "refreshGeneration"
    )
    assert "build_refresh_restored" in validation
    assert "putSettings" in index_manager
    assert "acknowledged" in index_manager
    assert "refresh_interval" in index_manager
    assert "=== '1s'" in index_manager
    assert "refresh_interval_restored" in activation
    assert "generation_refresh_state_failed" in metrics
    assert "build_refresh_interval" in metrics
    assert "normal_refresh_interval" in metrics
    assert "refresh_state" in metrics
    assert "BuildRefreshService::class" in integration
    assert "ValidationService::class" in integration
    assert "refresh_state" in integration


def test_opensearch_integration_qualifies_base64_vector_parity() -> None:
    integration = (
        ROOT / "tests/test_module_opensearch_hybrid_opensearch.py"
    ).read_text()
    workflow = (ROOT / ".github/workflows/module-opensearch-hybrid.yml").read_text()

    assert "struct.pack(\"<256f\"" in integration
    assert "base64.b64encode" in integration
    assert '"docvalue_fields": ["embedding"]' in integration
    assert '"script_score"' in integration
    assert '"/_bulk",' in integration
    assert "not-valid-base64" in integration
    assert "numeric_hits == base64_hits" in integration
    assert "Qualify OpenSearch 3.8 vector wire parity" in workflow
    assert "RUN_OPENSEARCH_INTEGRATION: '1'" in workflow
    assert "tests/test_module_opensearch_hybrid_opensearch.py" in workflow


def test_fake_generations_are_nonactivatable_at_two_boundaries() -> None:
    build_command = (MODULE / "Console/BuildCommand.php").read_text()
    activation = (MODULE / "Model/Activation/ActivationService.php").read_text()

    assert "Use deterministic development vectors that can never be activated" in build_command
    assert "if ((bool)$generation['is_fake'])" in activation


def test_admin_configuration_paths_match_runtime_scope_paths() -> None:
    system = _xml("etc/adminhtml/system.xml")
    section = system.find("system/section[@id='mageos_opensearch_hybrid']")
    catalog = system.find("system/section[@id='catalog']")

    assert section is not None
    tab = section.find("tab")
    resource = section.find("resource")
    assert tab is not None
    assert resource is not None
    assert tab.text == "catalog"
    assert resource.text == "MageOS_OpenSearchHybrid::config"
    assert section.find("group[@id='general']/field[@id='enabled']") is not None
    assert section.find("group[@id='encoder']/field[@id='batch_size']") is not None
    assert catalog is not None
    assert catalog.find("group[@id='search']") is not None


def test_admin_operations_page_has_consistent_route_menu_acl_and_one_column_layout() -> None:
    routes = _xml("etc/adminhtml/routes.xml")
    menu = _xml("etc/adminhtml/menu.xml")
    layout = _xml("view/adminhtml/layout/mageos_opensearch_hybrid_status_index.xml")
    controller = (MODULE / "Controller/Adminhtml/Status/Index.php").read_text()
    preview = (MODULE / "Controller/Adminhtml/Status/Preview.php").read_text()
    template = (MODULE / "view/adminhtml/templates/status.phtml").read_text()

    route = routes.find("router[@id='admin']/route[@id='mageos_opensearch_hybrid']")
    item = menu.find("menu/add[@id='MageOS_OpenSearchHybrid::status']")
    assert route is not None
    assert route.attrib["frontName"] == "mageos_opensearch_hybrid"
    assert item is not None
    assert item.attrib["action"] == "mageos_opensearch_hybrid/status/index"
    assert item.attrib["resource"] == "MageOS_OpenSearchHybrid::status"
    assert layout.attrib["layout"] == "admin-1column"
    assert "MageOS_OpenSearchHybrid::status" in controller
    assert "HttpPostActionInterface" in preview
    assert "getBlockHtml('formkey')" in template
    assert "estimated_documents" in template

    runner = (MODULE / "dev/ci/run.sh").read_text()
    assert "COMPOSER_MIRROR_PATH_REPOS=1" in runner
    assert "assert-admin-status-render.php" in runner
    assert "assert-doctor.php" in runner


def test_admin_activation_and_rollback_are_acl_scoped_previewed_post_mutations() -> None:
    acl = _xml("etc/acl.xml")
    preview = (MODULE / "Controller/Adminhtml/Activation/Preview.php").read_text()
    apply = (MODULE / "Controller/Adminhtml/Activation/Apply.php").read_text()
    block = (MODULE / "Block/Adminhtml/Status.php").read_text()
    template = (MODULE / "view/adminhtml/templates/status.phtml").read_text()
    activation = (MODULE / "Model/Activation/ActivationService.php").read_text()

    assert acl.find(
        ".//resource[@id='MageOS_OpenSearchHybrid::activate']"
    ) is not None
    for controller in (preview, apply):
        assert "HttpPostActionInterface" in controller
        assert "MageOS_OpenSearchHybrid::activate" in controller
        assert "catch (LocalizedException|\\InvalidArgumentException $exception)" in controller
        assert "LocalizedException|\\InvalidArgumentException|\\RuntimeException" not in controller
        assert "'error_class' => $throwable::class" in controller
    assert "canManageActivation" in block
    assert "getActivationPreviewUrl" in block
    assert "getActivationApplyUrl" in block
    assert template.count("getBlockHtml('formkey')") >= 5
    assert "Exact generation evidence" in template
    assert "Exact validation result" in template
    assert 'name="human_confirmation"' in template
    assert 'name="confirmation_token"' in template
    assert "previewActivation" in activation
    assert "previewNativeRollback" in activation
    assert "previewGenerationRollback" in activation
    assert "activateConfirmed" in activation
    assert "rollbackToNativeConfirmed" in activation
    assert "rollbackToGenerationConfirmed" in activation
    assert "admin_user_id:" in apply
    assert "'actor' => $this->normalizeActor($actor)" in activation


def test_full_build_seeding_is_resumable_bounded_and_keyset_paginated() -> None:
    schema = _xml("etc/db_schema.xml")
    progress = schema.find("table[@name='mageos_opensearch_hybrid_generation_progress']")
    build = (MODULE / "Model/Generation/BuildService.php").read_text()
    build_command = (MODULE / "Console/BuildCommand.php").read_text()
    outbox = (MODULE / "Model/Outbox/OutboxRepository.php").read_text()
    reconciliation = (MODULE / "Model/Outbox/ReconciliationService.php").read_text()
    resolver = (MODULE / "Model/Change/SearchableProductResolver.php").read_text()

    assert progress is not None
    columns = {column.attrib["name"] for column in progress.findall("column")}
    assert {"last_seeded_product_id", "seeded_products", "seeding_state"} <= columns
    assert "public function resumeFake" in build
    assert "public function pauseFake" in build
    assert "nextEligibleIds" in build
    assert "countEligible" in build
    assert "'gt' => $afterProductId" in resolver
    assert "getAllIds($batchSize)" in resolver
    assert "limit($batchSize, $offset)" not in build
    assert "countOutstandingForGeneration" in build
    assert "maxOutstandingBatches" in build
    assert "countOutstandingForGeneration" in outbox
    assert "resumePendingBuilds" in reconciliation
    assert "'resume'," in build_command
    assert "'pause'," in build_command
    assert "buildService->resume(" in build_command
    assert "buildService->pause(" in build_command
    assert "'PAUSED'" in build
    assert build.index("$connection->beginTransaction();") < build.index(
        "$connection->insert($generationTable"
    )
    assert build.index("$connection->insert($progressTable") < build.index(
        "$connection->commit();"
    )
    assert "$connection->rollBack();" in build


def test_fake_encoder_is_explicitly_ineligible_and_has_a_pinned_base_image() -> None:
    encoder = (ENCODER / "src/mageos_opensearch_hybrid_encoder/encoder.py").read_text()
    server = (ENCODER / "src/mageos_opensearch_hybrid_encoder/server.py").read_text()
    dockerfile = (ENCODER / "Dockerfile").read_text()

    assert 'RUNTIME = "deterministic_hash_fake_only"' in encoder
    assert '"production_eligible": False' in server
    assert dockerfile.splitlines()[0].startswith("FROM python:3.13.7-alpine3.22@sha256:")


def test_production_encoder_packaging_is_registry_neutral() -> None:
    dockerfile = (ENCODER / "Dockerfile.production").read_text()
    compose = (ENCODER / "deploy/compose.yaml").read_text()
    docker_network = (ENCODER / "deploy/compose.docker-network.yaml").read_text()
    loopback = (ENCODER / "deploy/compose.loopback.yaml").read_text()
    environment = (ENCODER / "deploy/encoder.env.example").read_text()

    assert "ARG PYTHON_BASE_IMAGE" in dockerfile
    assert "ARG SOURCE_URL" in dockerfile
    assert 'org.opencontainers.image.source="${SOURCE_URL}"' in dockerfile
    assert 'test -n "${SOURCE_URL}"' in dockerfile
    assert 'test -n "${SOURCE_REVISION}"' in dockerfile
    assert "rocketweb" not in dockerfile
    assert "FROM ${PYTHON_BASE_IMAGE}" in dockerfile
    assert "requirements-production.lock" in dockerfile
    assert "release/artifacts" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "registry.example/namespace/opensearch-hybrid-encoder@sha256:" in environment
    assert "ENCODER_DEPLOYMENT_DIGEST=sha256:" in environment
    assert "ENCODER_PULL_POLICY=always" in environment
    assert "ENCODER_API_TOKEN_FILE: /run/secrets/encoder_api_token" in compose
    assert "ENCODER_EXPECTED_DEPLOYMENT_DIGEST" in compose
    assert "pull_policy: ${ENCODER_PULL_POLICY" in compose
    assert "read_only: true" in compose
    assert "cap_drop:" in compose
    assert "no-new-privileges:true" in compose
    assert "ports:" not in compose
    assert "ports:" not in docker_network
    assert "external: true" in docker_network
    assert "ENCODER_NETWORK_ALIAS" in docker_network
    assert "127.0.0.1:${ENCODER_LOOPBACK_PORT" in loopback

    workflow = (ROOT / ".github/workflows/module-opensearch-hybrid.yml").read_text()
    assert "  pull_request:\n  push:\n" in workflow
    assert "encoder-unit:" in workflow
    assert "requirements-production.lock" in workflow
    assert "pip-audit==2.10.1" in workflow
    assert "--require-hashes --disable-pip --strict --progress-spinner=off" in workflow
    assert "run: make check" in workflow


def test_confirmed_config_invalidations_create_fail_closed_replacement_generations() -> None:
    service = (MODULE / "Model/Generation/AutomaticReplacementService.php").read_text()
    repository = (MODULE / "Model/Generation/ReplacementCandidateRepository.php").read_text()
    reconciliation = (MODULE / "Model/Outbox/ReconciliationService.php").read_text()

    assert "withinStoreLock" in service
    assert "isEnabled" in service
    assert "isStoreIncluded" in service
    assert "buildProduction" in service
    assert "FAILED_TO_CREATE" in service
    assert "error_class" in service
    assert "activate" not in service
    assert "GET_LOCK" in repository
    assert "RELEASE_LOCK" in repository
    assert "entity_type = ?" in repository
    assert "'CONFIGURATION'" in repository
    assert "replacement_already_created" in repository
    assert "replacementService->createPending($storeId)" in reconciliation
    assert "'replacements' => $replacements" in reconciliation
    assert "'failed_replacements' => $failedReplacements" in reconciliation


def test_production_encoder_has_reproducible_amd64_qualification_commands() -> None:
    qualification = (
        ENCODER / "src/mageos_opensearch_hybrid_encoder/qualification.py"
    ).read_text()
    release = (ENCODER / "src/mageos_opensearch_hybrid_encoder/release.py").read_text()
    runtime = (ENCODER / "src/mageos_opensearch_hybrid_encoder/runtime.py").read_text()
    readme = (ENCODER / "README.md").read_text()

    assert '"capture-reference"' in qualification
    assert '"verify-amd64"' in qualification
    assert '"measure-latency"' in qualification
    assert "PARITY_MAXIMUM_ABSOLUTE_DELTA = 1e-5" in qualification
    assert "PARITY_MINIMUM_COSINE_SIMILARITY = 0.999999" in qualification
    assert "load_qualification_artifacts" in runtime
    assert '"evidence/amd64-parity.json"' in release
    assert '"evidence/latency-concurrency-four.json"' in release
    assert "EXPECTED_RUNTIME_VERSIONS" in release
    assert "qualification capture-reference" in readme
    assert "qualification verify-amd64" in readme
    assert "qualification measure-latency" in readme
    assert "mageos_opensearch_hybrid_encoder.release" in readme


def test_product_changes_are_captured_before_native_index_commit_acknowledgement() -> None:
    schema = _xml("etc/db_schema.xml")
    di = _xml("etc/di.xml")
    journal = schema.find("table[@name='mageos_opensearch_hybrid_change_journal']")
    product_type = di.find("type[@name='Magento\\Catalog\\Model\\ResourceModel\\Product']")
    fulltext_type = di.find("type[@name='Magento\\CatalogSearch\\Model\\Indexer\\Fulltext']")
    capture = (MODULE / "Model/Change/CaptureService.php").read_text()
    product_plugin = (MODULE / "Plugin/ProductChangeCapture.php").read_text()
    native_plugin = (MODULE / "Plugin/NativeIndexerAcknowledgement.php").read_text()
    journal_repository = (MODULE / "Model/Change/ChangeJournalRepository.php").read_text()
    priority_consumer = (MODULE / "Model/Queue/PriorityConsumer.php").read_text()
    priority_processor = (MODULE / "Model/Queue/PriorityJobProcessor.php").read_text()

    assert journal is not None
    columns = {column.attrib["name"] for column in journal.findall("column")}
    assert {
        "native_state",
        "hybrid_state",
        "native_completed_at",
        "hybrid_completed_at",
    } <= columns

    assert product_type is not None
    product_registration = product_type.find(
        "plugin[@name='mageos_opensearch_hybrid_product_capture']"
    )
    assert product_registration is not None
    assert product_registration.attrib["type"] == (
        "MageOS\\OpenSearchHybrid\\Plugin\\ProductChangeCapture"
    )
    assert product_registration.attrib["sortOrder"] == "10"

    assert fulltext_type is not None
    native_registration = fulltext_type.find(
        "plugin[@name='mageos_opensearch_hybrid_native_acknowledgement']"
    )
    assert native_registration is not None
    assert native_registration.attrib["type"] == (
        "MageOS\\OpenSearchHybrid\\Plugin\\NativeIndexerAcknowledgement"
    )

    assert "addCommitCallback" in product_plugin
    assert "aroundSave" in product_plugin
    assert "aroundDelete" in product_plugin
    assert "captureProduct" in capture
    assert "requireWatermark" in capture
    assert "writableForStore" in capture
    assert "supersedeProductJobs" in capture
    assert "supersedeProductJobs" in (MODULE / "Model/Outbox/OutboxRepository.php").read_text()
    assert "aroundExecuteByDimensions" in native_plugin
    assert "acknowledgeNative" in native_plugin
    assert "safeWatermark" in journal_repository
    assert "acknowledgeHybridJob" in priority_consumer
    assert "acknowledgeHybridJob" not in priority_processor
    assert "advanceHybrid" not in priority_processor

    assert (
        "$subject->beginTransaction();\n        try {\n            $result = $proceed($product);"
        in product_plugin
    )
    assert product_plugin.index("captureProduct") < product_plugin.index("$subject->commit()")
    assert "$subject->rollBack()" in product_plugin


def test_delete_tombstones_embedding_state_and_decrements_generation_coverage_once() -> None:
    interface = (MODULE / "Api/EmbeddingStoreInterface.php").read_text()
    store = (MODULE / "Model/Embedding/EmbeddingStore.php").read_text()
    priority = (MODULE / "Model/Queue/PriorityJobProcessor.php").read_text()

    assert "public function delete(" in interface
    assert "public function delete(" in store
    assert "'status' => 'DELETED'" in store
    assert "coverage_complete" in store
    assert "$this->embeddingStore->delete(" in priority


def test_bulk_attribute_and_website_scope_changes_are_captured_atomically() -> None:
    bulk = (MODULE / "Plugin/BulkAttributeChangeCapture.php").read_text()
    website = (MODULE / "Plugin/ProductWebsiteChangeCapture.php").read_text()
    product = (MODULE / "Plugin/ProductChangeCapture.php").read_text()
    resolver = (MODULE / "Model/Change/ProductStoreResolver.php").read_text()
    di = _xml("etc/di.xml")

    assert "$subject->beginTransaction();" in bulk
    assert "$subject->addCommitCallback(" in bulk
    assert "product_attribute_bulk_update" in bulk
    assert "$connection->beginTransaction();" in website
    assert "$connection->commit();" in website
    assert "product_website_add" in website
    assert "product_website_remove" in website
    assert "resolveWebsiteStores" in website
    assert "resolveWebsiteStores" in resolver
    assert "$previousStoreIds" in product
    assert "array_unique(array_merge($previousStoreIds, $currentStoreIds))" in product

    assert di.find(
        "type[@name='Magento\\Catalog\\Model\\ResourceModel\\Product\\Action']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\BulkAttributeChangeCapture']"
    ) is not None
    assert di.find(
        "type[@name='Magento\\Catalog\\Model\\Product\\Website']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\ProductWebsiteChangeCapture']"
    ) is not None


def test_salability_changes_use_priority_only_refresh_and_preserve_only_matching_vectors() -> None:
    contract = json.loads((MODULE / "etc/opensearch_hybrid_contract.json").read_text())
    document = (MODULE / "Model/Document/ProductDocumentFactory.php").read_text()
    capture = (MODULE / "Model/Change/CaptureService.php").read_text()
    plugin = (MODULE / "Plugin/InventorySalabilityChangeCapture.php").read_text()
    source_plugin = (MODULE / "Plugin/SourceItemsSaveCapture.php").read_text()
    context = (MODULE / "Model/Change/InventoryCaptureContext.php").read_text()
    priority = (MODULE / "Model/Queue/PriorityJobProcessor.php").read_text()
    embedding = (MODULE / "Model/Embedding/EmbeddingStore.php").read_text()
    composer = json.loads((MODULE / "composer.json").read_text())
    module_xml = _xml("etc/module.xml")
    di = _xml("etc/di.xml")

    properties = contract["mapping"]["properties"]
    assert properties["stock_id"] == {"type": "integer"}
    assert properties["is_salable"] == {"type": "boolean"}
    assert "GetStockIdForByStoreId" in document
    assert "AreProductsSalableInterface" in document
    assert "'stock_id' => $stockId" in document
    assert "'is_salable' => $isSalable" in document
    assert "capturePriorityProductsByStore" in capture
    assert "'REFRESH'" in capture
    assert "capturePriorityProductsByStore" in plugin
    assert "capturePriorityProductsByStore" in source_plugin
    assert "$connection->beginTransaction();" in source_plugin
    assert "$connection->commit();" in source_plugin
    assert "$connection->rollBack();" in source_plugin
    assert "inventory_salability_change" in plugin
    assert "inventory_salability_change" in source_plugin
    assert "enterSourceItemsSave" in source_plugin
    assert "leaveSourceItemsSave" in source_plugin
    assert "isSourceItemsSave" in plugin
    assert "isSourceItemsSave" in context
    assert "getVectorForSource" in priority
    assert "getVectorForSource" in embedding

    required = composer["require"]
    assert required["mage-os/module-inventory-catalog"] == "3.4.*"
    assert required["mage-os/module-inventory-catalog-api"] == "3.4.*"
    assert required["mage-os/module-inventory-catalog-search"] == "3.4.*"
    assert required["mage-os/module-inventory-indexer"] == "3.4.*"
    assert required["mage-os/module-inventory-sales-api"] == "3.4.*"
    sequence = {
        item.attrib["name"] for item in module_xml.findall("module/sequence/module")
    }
    assert {
        "Magento_InventoryCatalog",
        "Magento_InventoryCatalogSearch",
        "Magento_InventoryIndexer",
    } <= sequence
    source_capture = di.find(
        "type[@name='Magento\\InventoryApi\\Api\\SourceItemsSaveInterface']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\SourceItemsSaveCapture']"
    )
    assert source_capture is not None
    assert source_capture.attrib["sortOrder"] == "-100"
    assert di.find(
        "type[@name='Magento\\InventoryCatalogSearch\\Model\\UpdateFulltextIndexOnProductSalabilityChange']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\InventorySalabilityChangeCapture']"
    ) is not None


def test_price_index_rows_use_store_customer_group_scope_and_priority_only_capture() -> None:
    contract = json.loads((MODULE / "etc/opensearch_hybrid_contract.json").read_text())
    document = (MODULE / "Model/Document/ProductDocumentFactory.php").read_text()
    resolver = (MODULE / "Model/Document/PriceIndexResolver.php").read_text()
    plugin = (MODULE / "Plugin/PriceIndexChangeCapture.php").read_text()
    journal = (MODULE / "Model/Change/ChangeJournalRepository.php").read_text()
    di = _xml("etc/di.xml")

    prices = contract["mapping"]["properties"]["customer_group_prices"]
    assert prices["type"] == "nested"
    assert prices["properties"]["customer_group_id"] == {"type": "integer"}
    assert prices["properties"]["website_id"] == {"type": "integer"}
    assert prices["properties"]["regular_price"] == {"type": "double"}
    assert prices["properties"]["final_price"] == {"type": "double"}
    assert prices["properties"]["min_price"] == {"type": "double"}
    assert prices["properties"]["max_price"] == {"type": "double"}
    assert prices["properties"]["tier_price"] == {"type": "double"}
    assert "PriceIndexResolver" in document
    assert "customer_group_prices" in document
    assert "catalog_product_index_price" in resolver
    assert "customer_group_id" in resolver
    assert "website_id" in resolver
    assert "PriceTableResolver" in resolver
    assert "DimensionModeConfiguration" in resolver
    assert "CustomerGroupDimensionProvider::DIMENSION_NAME" in resolver
    assert "WebsiteDimensionProvider::DIMENSION_NAME" in resolver
    assert "new \\Magento\\Framework\\Indexer\\Dimension" in resolver
    assert "capturePriorityProductsByStore" in plugin
    assert "product_price_index" in plugin
    assert "aroundExecuteList" in plugin
    assert "aroundExecuteRow" in plugin
    assert "aroundExecuteFull" in plugin
    assert "nextEligibleIds" in plugin
    assert "batchSize" in plugin
    assert "product_price_full_reindex" in plugin
    assert "acknowledgeNativeChanges" in plugin
    assert "acknowledgeNativeChanges" in journal
    assert di.find(
        "type[@name='Magento\\Catalog\\Model\\Indexer\\Product\\Price']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\PriceIndexChangeCapture']"
    ) is not None


def test_native_partial_reindex_repairs_dependent_changes_without_duplicate_capture() -> None:
    capture = (MODULE / "Model/Change/CaptureService.php").read_text()
    journal = (MODULE / "Model/Change/ChangeJournalRepository.php").read_text()
    resolver = (MODULE / "Model/Change/SearchableProductResolver.php").read_text()
    native_plugin = (MODULE / "Plugin/NativeIndexerAcknowledgement.php").read_text()

    assert "captureNativeProducts" in capture
    assert "untrackedNativeProductIds" in capture
    assert "untrackedNativeProductIds" in journal
    assert "eligibleIds" in capture
    assert "addStoreFilter" in resolver
    assert "'status'," in resolver
    assert "getVisibleInSearchIds" in resolver
    assert "captureNativeProducts" in native_plugin
    assert native_plugin.index("captureNativeProducts") < native_plugin.index(
        "pendingNativeBoundary"
    )
    assert "native_partial_reindex" in native_plugin


def test_category_names_and_composite_parent_text_have_explicit_invalidation_paths() -> None:
    di = _xml("etc/di.xml")
    category_type = di.find("type[@name='Magento\\Catalog\\Model\\ResourceModel\\Category']")
    category_plugin = (MODULE / "Plugin/CategoryChangeCapture.php").read_text()
    category_resolver = (MODULE / "Model/Change/CategoryProductResolver.php").read_text()
    category_text = (MODULE / "Model/Document/CategoryTextResolver.php").read_text()
    document = (MODULE / "Model/Document/ProductDocumentFactory.php").read_text()
    native_plugin = (MODULE / "Plugin/NativeIndexerAcknowledgement.php").read_text()

    assert category_type is not None
    registration = category_type.find(
        "plugin[@name='mageos_opensearch_hybrid_category_capture']"
    )
    assert registration is not None
    assert registration.attrib["type"] == "MageOS\\OpenSearchHybrid\\Plugin\\CategoryChangeCapture"
    assert "dataHasChangedFor('name')" in category_plugin
    assert "descendantProductIds" in category_plugin
    assert "captureProductsByStore" in category_plugin
    assert "catalog_category_product" in category_resolver
    assert "category.path LIKE" in category_resolver
    assert "CategoryTextResolver" in document
    assert "path" in category_text
    assert "getRelationsByChild" in native_plugin
    assert (
        "$subject->beginTransaction();\n        try {\n            $result = $proceed($category);"
        in category_plugin
    )
    assert category_plugin.index("captureProductsByStore") < category_plugin.index(
        "$subject->commit()"
    )
    assert "$subject->rollBack()" in category_plugin


def test_frozen_attribute_labels_feed_the_correct_lexical_and_semantic_fields() -> None:
    contract = json.loads((MODULE / "etc/opensearch_hybrid_contract.json").read_text())
    registry = (MODULE / "Model/Document/SemanticAttributeRegistry.php").read_text()
    resolver = (MODULE / "Model/Document/AttributeTextResolver.php").read_text()
    document = (MODULE / "Model/Document/ProductDocumentFactory.php").read_text()

    assert contract["model"]["lexical_brand_attributes"] == ["manufacturer"]
    assert contract["model"]["semantic_feature_attributes"] == ["color", "material"]
    assert "lexical_brand_attributes" in registry
    assert "semantic_feature_attributes" in registry
    assert "getFrontend()->getValue($product)" in resolver
    assert "$attribute->setStoreId($storeId)" in resolver
    assert "html_entity_decode" in resolver
    assert "strip_tags" in resolver
    assert "AttributeTextResolver" in document
    assert "'brand' => $attributeText['brand']" in document
    assert "'features' => $attributeText['features']" in document
    assert "array_merge(" in document
    assert "$attributeText['semantic']" in document


def test_mapped_option_label_changes_capture_affected_products_atomically() -> None:
    plugin = (MODULE / "Plugin/AttributeOptionChangeCapture.php").read_text()
    resolver = (MODULE / "Model/Change/AttributeOptionProductResolver.php").read_text()
    di = _xml("etc/di.xml")

    assert "aroundSave" in plugin
    assert "getOption()" in plugin
    assert "changedStoreIds" in plugin
    assert "captureProductsByStore" in plugin
    assert "attribute_option_label_change" in plugin
    assert "$subject->beginTransaction();" in plugin
    assert "$subject->addCommitCallback(" in plugin
    assert "$subject->rollBack();" in plugin
    assert "eav_attribute_option_value" in resolver
    assert "prepareSqlCondition('value', ['finset'" in resolver
    assert "resolveMany" in resolver
    assert di.find(
        "type[@name='Magento\\Eav\\Model\\ResourceModel\\Entity\\Attribute']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\AttributeOptionChangeCapture']"
    ) is not None


def test_newer_product_work_coalesces_older_jobs_without_losing_journal_coverage() -> None:
    capture = (MODULE / "Model/Change/CaptureService.php").read_text()
    outbox = (MODULE / "Model/Outbox/OutboxRepository.php").read_text()
    journal = (MODULE / "Model/Change/ChangeJournalRepository.php").read_text()

    assert "supersedeProductJobs" in capture
    assert "public function supersedeProductJobs" in outbox
    assert "MIN(job.earliest_change_id)" in outbox
    assert "[self::COMPLETED, 'SUPERSEDED']" in journal
    assert "min(" in capture
    assert "$priorityEarliestChangeId" in capture
    assert "if ($earliestChangeId <= 0)" not in outbox
    assert "if ((string)$item['state'] === 'SUPERSEDED')" in (
        MODULE / "Model/Queue/PriorityJobProcessor.php"
    ).read_text()


def test_generation_validation_treats_superseded_work_as_resolved() -> None:
    validation = (MODULE / "Model/Generation/ValidationService.php").read_text()
    activation = (MODULE / "Model/Activation/ActivationService.php").read_text()

    assert "state NOT IN (?)" in validation
    assert "['COMPLETED', 'SUPERSEDED']" in validation
    assert "state NOT IN (?)" in activation
    assert "['COMPLETED', 'SUPERSEDED']" in activation


def test_dual_generation_catch_up_invalidates_ready_state_and_resnapshots_coverage() -> None:
    generation = (MODULE / "Model/Generation/GenerationRepository.php").read_text()
    capture = (MODULE / "Model/Change/CaptureService.php").read_text()
    validation = (MODULE / "Model/Generation/ValidationService.php").read_text()
    index_manager = (MODULE / "Model/OpenSearch/IndexManager.php").read_text()

    assert "['ACTIVE', 'BUILDING', 'CATCHING_UP', 'READY']" in generation
    assert "->forUpdate(true)" in capture
    assert "'state' => 'CATCHING_UP'" in capture
    assert "'validation_report' => null" in capture
    assert "countEligible" in validation
    assert "captured_change_id" in validation
    assert "coverage_total" in validation
    assert "native_state" in validation
    assert "hybrid_state" in validation
    assert "catch_up_stable" in validation
    assert "result_contract_accepted" in validation
    assert "refreshGeneration" in validation
    assert "public function refreshGeneration" in index_manager


def test_generation_scope_changes_require_a_replacement_build() -> None:
    schema = (MODULE / "etc" / "db_schema.xml").read_text()
    di = (MODULE / "etc" / "di.xml").read_text()
    module = (MODULE / "etc" / "module.xml").read_text()
    invalidation = (MODULE / "Model" / "Change" / "ScopeInvalidationService.php").read_text()
    journal = (MODULE / "Model" / "Change" / "ChangeJournalRepository.php").read_text()
    validation = (MODULE / "Model" / "Generation" / "ValidationService.php").read_text()
    activation = (MODULE / "Model" / "Activation" / "ActivationService.php").read_text()

    assert 'name="scope_digest" nullable="true"' in schema
    assert "Magento\\Config\\Model\\ResourceModel\\Config\\Data" in di
    assert "Magento\\Store\\Model\\ResourceModel\\Store" in di
    assert "Magento\\Store\\Model\\ResourceModel\\Website" in di
    assert "Magento\\Store\\Model\\ResourceModel\\Group" in di
    assert '<module name="Magento_Store"/>' in module
    assert "appendGenerationInvalidation" in invalidation
    assert "requireWatermark" in invalidation
    assert "'state' => 'FAILED'" in invalidation
    assert "supersedeGenerationJobs" in invalidation
    assert "acknowledgeGenerationInvalidations" in journal
    assert "entity_type = 'PRODUCT'" in validation
    assert "scope_digest" in validation
    assert "acknowledgeGenerationInvalidations" in activation


def test_active_generation_must_match_the_current_retrieval_contract() -> None:
    generation = (MODULE / "Model" / "Generation" / "GenerationRepository.php").read_text()
    metrics = (MODULE / "Model" / "Operations" / "OperationalMetricsService.php").read_text()

    assert "matchesCurrentContract" in generation
    assert "$this->retrievalContract->digest()" in generation
    assert "$this->retrievalContract->resultContractDigest()" in generation
    assert "$this->retrievalContract->mappingDigest()" in generation
    assert "$this->retrievalContract->pipelineDigest()" in generation
    assert "'active_for_store' => $activeForStore" in metrics
    assert "'contract_current' => $contractCurrent" in metrics
    assert "active_generation_contract_drift" in metrics


def test_query_telemetry_uses_a_dedicated_privacy_safe_log_channel() -> None:
    di = _xml("etc/di.xml")
    wiring = di.find(
        "type[@name='MageOS\\OpenSearchHybrid\\Model\\Operations\\QueryTelemetry']"
    )
    handler = (MODULE / "Logger/Handler/QueryTelemetry.php").read_text()
    logger = (MODULE / "Logger/QueryTelemetry.php").read_text()
    span = (MODULE / "Model/Operations/QueryTelemetrySpan.php").read_text()
    operations = (MODULE / "docs/OPERATIONS.md").read_text()

    assert wiring is not None
    assert wiring.findtext("arguments/argument[@name='logger']") == (
        "MageOS\\OpenSearchHybrid\\Logger\\QueryTelemetry"
    )
    assert "extends Monolog" in logger
    assert "parent::__construct('mageos_opensearch_hybrid_query', [$handler])" in logger
    assert "QueryTelemetryLogger" not in (MODULE / "etc/di.xml").read_text()
    assert "'/var/log/opensearch-hybrid-query.log'" in handler
    assert "Logger::INFO" in handler
    assert "$context['error_class'] = $failure::class" in span
    assert "getMessage()" not in span
    assert "$queryText" not in span
    assert "var/log/opensearch-hybrid-query.log" in operations


def test_generation_scope_digest_covers_store_and_generation_affecting_config() -> None:
    fingerprint = (MODULE / "Model" / "Generation" / "StoreScopeFingerprint.php").read_text()
    build = (MODULE / "Model" / "Generation" / "BuildService.php").read_text()
    config_capture = (MODULE / "Plugin" / "ConfigurationChangeCapture.php").read_text()
    config_impact = (MODULE / "Model" / "Change" / "ConfigurationImpactService.php").read_text()
    store_capture = (MODULE / "Plugin" / "StoreDefinitionChangeCapture.php").read_text()

    assert "website_id" in fingerprint
    assert "group_id" in fingerprint
    assert "root_category_id" in fingerprint
    assert "is_active" in fingerprint
    assert "general/locale/code" in fingerprint
    assert "currency/options/base" in fingerprint
    assert "catalog/price/scope" in fingerprint
    assert "XML_PATH_INCLUDED_STORES" in fingerprint
    assert "XML_PATH_ENCODER_ENDPOINT" in fingerprint
    assert (
        "'scope_digest' => $scopeDigest ?? $this->storeScopeFingerprint->digest($storeId)"
        in build
    )
    assert "GENERATION_AFFECTING_PATHS" in config_impact
    assert "ConfigurationImpactService" in config_capture
    assert "afterSave" in config_capture
    assert "CONFIGURATION" in config_capture
    assert "instanceof \\Magento\\Store\\Model\\Store" in store_capture
    assert "instanceof \\Magento\\Store\\Model\\Website" in store_capture
    assert "instanceof \\Magento\\Store\\Model\\Group" in store_capture


def test_outbox_failures_reach_a_bounded_dead_state_and_manual_replay_is_audited() -> None:
    schema = _xml("etc/db_schema.xml")
    outbox_table = schema.find("table[@name='mageos_opensearch_hybrid_outbox']")
    config = (MODULE / "Model/Config.php").read_text()
    defaults = _xml("etc/config.xml")
    system = _xml("etc/adminhtml/system.xml")
    outbox = (MODULE / "Model/Outbox/OutboxRepository.php").read_text()
    embedding_consumer = (MODULE / "Model/Queue/EmbeddingConsumer.php").read_text()
    priority_consumer = (MODULE / "Model/Queue/PriorityConsumer.php").read_text()

    assert outbox_table is not None
    columns = {column.attrib["name"] for column in outbox_table.findall("column")}
    assert "replay_count" in columns
    assert "XML_PATH_JOB_MAX_ATTEMPTS" in config
    assert "jobMaxAttempts" in config
    attempts = defaults.find(
        "default/mageos_opensearch_hybrid/reliability/job_max_attempts"
    )
    assert attempts is not None
    assert attempts.text == "5"
    reliability = system.find(
        "system/section[@id='mageos_opensearch_hybrid']/group[@id='reliability']"
    )
    assert reliability is not None
    assert reliability.find("field[@id='job_max_attempts']") is not None
    assert "jobMaxAttempts" in outbox
    assert "'DEAD'" in outbox
    assert "attempt_count" in outbox
    assert "replay_count + 1" in outbox
    assert "if ($this->outboxRepository->retry" in embedding_consumer
    assert "if ($this->outboxRepository->retry" in priority_consumer


def test_inventory_changes_expand_composite_parents_and_capture_reservation_queue_updates() -> None:
    di = _xml("etc/di.xml")
    resolver = (MODULE / "Model/Change/InventoryProductResolver.php").read_text()
    source_capture = (MODULE / "Plugin/SourceItemsSaveCapture.php").read_text()
    reservation_capture = (
        MODULE / "Plugin/ReservationSalabilityChangeCapture.php"
    ).read_text()

    reservation_type = di.find(
        ".//type[@name='Magento\\InventoryIndexer\\Model\\Queue\\UpdateIndexSalabilityStatus']"
    )
    assert reservation_type is not None
    assert reservation_type.find(
        "plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\ReservationSalabilityChangeCapture']"
    ) is not None
    assert "GetParentSkusOfChildrenSkusInterface" in resolver
    assert "while ($frontier !== [])" in resolver
    assert "getStockIdForStore" in resolver
    assert "InventoryProductResolver" in source_capture
    assert "inventory_reservation_change" in reservation_capture
    assert "getStock()" in reservation_capture


def test_inventory_topology_is_frozen_and_fails_generations_closed() -> None:
    di = _xml("etc/di.xml")
    fingerprint = (MODULE / "Model/Generation/StoreScopeFingerprint.php").read_text()
    service = (
        MODULE / "Model/Change/InventoryTopologyInvalidationService.php"
    ).read_text()
    runner = (MODULE / "dev/ci/run.sh").read_text()

    assert "'schema_version' => 2" in fingerprint
    assert "'inventory' => $this->inventoryScope" in fingerprint
    assert "inventory_stock_sales_channel" in fingerprint
    assert "inventory_source_stock_link" in fingerprint
    assert "inventory_source" in fingerprint
    assert "inventory_sales_channel_change" in service
    journal = (MODULE / "Model/Change/ChangeJournalRepository.php").read_text()
    assert "'INVENTORY_SALES_CHANNEL'" in journal
    assert "'INVENTORY_STOCK'" in journal
    assert "inventory_stock_source_links_save" in (
        MODULE / "Plugin/StockSourceLinksSaveTopologyCapture.php"
    ).read_text()
    assert "inventory_stock_source_links_delete" in (
        MODULE / "Plugin/StockSourceLinksDeleteTopologyCapture.php"
    ).read_text()
    assert "inventory_source_status_change" in (
        MODULE / "Plugin/SourceStatusTopologyCapture.php"
    ).read_text()
    assert di.find(
        ".//type[@name='Magento\\InventorySalesApi\\Model\\ReplaceSalesChannelsForStockInterface']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\StockSalesChannelTopologyCapture']"
    ) is not None
    assert "assert-inventory-topology-invalidation.php" in runner
    assert "prepare-active-readiness.php" in runner
    assert "assert-active-readiness.php" in runner


def test_admin_config_save_requires_an_exact_generation_impact_confirmation() -> None:
    admin_di = _xml("etc/adminhtml/di.xml")
    service = (MODULE / "Model/Change/ConfigurationSaveConfirmation.php").read_text()
    template = (
        MODULE / "view/adminhtml/templates/config/confirmation.phtml"
    ).read_text()
    javascript = (
        MODULE / "view/adminhtml/web/js/config-confirmation.js"
    ).read_text()

    assert admin_di.find(
        ".//type[@name='Magento\\Config\\Model\\Config']"
        "/plugin[@type='MageOS\\OpenSearchHybrid\\Plugin\\ConfigurationSaveConfirmation']"
    ) is not None
    assert "previous_digest" in service
    assert "proposed_digest" in service
    assert "config-save-" in service
    assert "previous_effective" not in template
    assert "proposed_effective" not in template
    assert "mageos_opensearch_hybrid_confirmation_token" in template
    assert "Preview generation impact" in template
    assert "form.serialize()" in javascript
    assert "preview.changes.map" in javascript
    assert "preview.stores.map" in javascript
    assert "assert-admin-config-save-confirmation.php" in (
        MODULE / "dev/ci/run.sh"
    ).read_text()


def test_admin_config_preview_distinguishes_validation_from_runtime_failures() -> None:
    controller = (MODULE / "Controller/Adminhtml/Config/Preview.php").read_text()

    assert "catch (\\InvalidArgumentException $exception)" in controller
    assert "setHttpResponseCode(400)" in controller
    assert "catch (\\Throwable $throwable)" in controller
    assert "setHttpResponseCode(500)" in controller
    assert "error_class' => $throwable::class" in controller
    assert "$throwable->getMessage()" not in controller


def test_generation_cleanup_does_not_persist_raw_exception_messages() -> None:
    service = (MODULE / "Model/Generation/GenerationCleanupService.php").read_text()

    assert "'last_error_class' => $throwable::class" in service
    assert "'last_diagnostic' => 'cleanup failed'" in service
    assert "$throwable->getMessage()" not in service
    assert "throw $throwable;" in service


def test_radial_product_similarity_is_isolated_disabled_and_profile_bound() -> None:
    config = _xml("etc/config.xml")
    di = _xml("etc/di.xml")
    system = _xml("etc/adminhtml/system.xml")
    service = (MODULE / "Model/Similarity/SimilarityService.php").read_text()
    builder = (MODULE / "Model/Similarity/RadialRequestBuilder.php").read_text()
    profiles = (MODULE / "Model/Similarity/ThresholdProfileRegistry.php").read_text()
    interface = (MODULE / "Api/SimilarityServiceInterface.php").read_text()

    assert config.findtext("default/mageos_opensearch_hybrid/similarity/enabled") == "0"
    preference = di.find(
        "preference[@for='MageOS\\OpenSearchHybrid\\Api\\SimilarityServiceInterface']"
    )
    assert preference is not None
    similarity_group = system.find(
        "system/section[@id='mageos_opensearch_hybrid']/group[@id='similarity']"
    )
    assert similarity_group is not None
    assert similarity_group.attrib["showInStore"] == "1"
    assert similarity_group.find("field[@id='enabled']") is not None
    assert "min_score" not in (MODULE / "etc/di.xml").read_text()
    assert "searchByProduct" in interface
    assert "isSimilarityEnabled" in service
    assert "activeForStore" in service
    assert "result_contract_accepted" in service
    assert "fallback" in service.lower()
    assert "min_score" in builder
    assert "'k'" not in builder
    for required_filter in (
        "store_id",
        "generation_id",
        "model_revision",
        "embedding_eligible",
        "status",
        "visibility",
        "is_salable",
    ):
        assert required_filter in builder
    assert "'_score' => 'desc'" in builder
    assert "'entity_id' => 'asc'" in builder
    assert "'_source' => false" in builder
    for identity_field in (
        "model_id",
        "model_revision",
        "dimension",
        "similarity",
        "source_recipe_version",
        "store_ids",
        "use_case",
        "judgment_set_sha256",
        "calibration_date",
        "primary_metric",
        "min_score",
    ):
        assert identity_field in profiles
