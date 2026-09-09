from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev" / "ci"
WORKFLOW = ROOT / ".github" / "workflows" / "module-opensearch-hybrid.yml"
MAKEFILE = ROOT / "Makefile"
SBOM_TOOL = ROOT / "dev" / "sbom" / "tool"
UPGRADE_RUNBOOK = ROOT / "docs" / "UPGRADE.md"
PHPCS_RULESET = ROOT / "phpcs.xml.dist"
STOREFRONT_ACCEPTANCE = FIXTURE / "assert-storefront-acceptance.php"


def test_fixture_pins_the_supported_service_lane() -> None:
    compose = (FIXTURE / "compose.yaml").read_text()

    assert (
        "mysql:8.4@sha256:b3b90af2a6552ae30c266fdb7d5dd55f3afb72404bb78d37fe8a23eb857fd3fb"
        in compose
    )
    assert (
        "rabbitmq:4.1-management@sha256:72914305cde66172bd85bc7cbede1edbbd16e5f1076af8b0d9afd3727c1e7980"
        in compose
    )
    assert (
        "opensearchproject/opensearch:3.8.0@sha256:"
        "bcc1797519726ceb6d651d4a3e60b7c30da91793914a8dfe75fd441d4f641509" in compose
    )
    assert '"127.0.0.1:13306:3306"' in compose
    assert '"127.0.0.1:35672:5672"' in compose
    assert '"127.0.0.1:19201:9200"' in compose
    assert "--log-bin-trust-function-creators=1" in compose
    assert "container_name:" not in compose
    assert compose.count("healthcheck:") == 3


def test_storefront_acceptance_proves_healthy_luma_requests_use_hybrid() -> None:
    probe = STOREFRONT_ACCEPTANCE.read_text()

    assert "$lumaHybridCallsBefore" in probe
    assert "product_list_order=relevance" in probe
    assert "$lumaHybridCallsAfter <= $lumaHybridCallsBefore" in probe
    assert "Luma hybrid search did not complete through the query encoder." in probe


def test_fixture_installs_public_mageos_and_verifies_the_module() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    defaults_probe = (FIXTURE / "assert-defaults.php").read_text()

    assert "set -euo pipefail" in runner
    assert 'module_install_root="${MAGEOS_MODULE_INSTALL_ROOT:-${module_root}}"' in runner
    assert '"${module_install_root}"' in runner
    assert "mage-os/project-community-edition" in runner
    assert '"3.4.0"' in runner
    assert "https://repo.mage-os.org/" in runner
    assert "hyva-themes" not in runner
    assert "mage-os/mageos-async-events:^4.0.8" in runner
    assert "MageOS_OpenSearchHybrid" in runner
    assert "--amqp-host=127.0.0.1" in runner
    assert "setup:upgrade --keep-generated" in runner
    assert "setup:di:compile" in runner
    assert "module:status MageOS_OpenSearchHybrid" in runner
    assert "assert-defaults.php" in runner
    assert "assert-atomic-change-rollback.php" in runner
    assert "MAGEOS_ROOT=" in runner
    assert "phpunit.xml.dist" in runner
    assert "mageos_opensearch_hybrid/general/enabled" in defaults_probe
    assert "ScopeConfigInterface::SCOPE_TYPE_DEFAULT" in defaults_probe


def test_fixture_proves_product_change_capture_rolls_back_with_the_catalog_write() -> None:
    probe = (FIXTURE / "assert-atomic-change-rollback.php").read_text()

    assert "CREATE TRIGGER" in probe
    assert "SIGNAL SQLSTATE" in probe
    assert "ProductRepositoryInterface::class" in probe
    assert "DROP TRIGGER" in probe
    assert "reload" in probe


def test_fixture_proves_correctness_queue_isolation_under_embedding_backlog() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    prepare_probe = (FIXTURE / "prepare-queue-isolation.php").read_text()
    assertion_probe = (FIXTURE / "assert-queue-isolation.php").read_text()

    assert "prepare-queue-isolation.php" in runner
    assert "queue:consumers:start mageos.opensearch_hybrid.correctness.consumer" in runner
    assert "--max-messages=1" in runner
    assert "assert-queue-isolation.php" in runner
    assert "correctness_elapsed" in runner
    assert "20" in runner
    assert runner.count("indexer:reindex catalogsearch_fulltext") == 2

    assert "BuildService::class" in prepare_probe
    assert "OutboxRepository::class" in prepare_probe
    assert "DataPlanePublisher::class" in prepare_probe
    assert "ProductRepositoryInterface::class" in prepare_probe
    assert "mageos_opensearch_hybrid_change_journal" in prepare_probe
    assert "native_state" in prepare_probe
    assert "'PENDING'" in prepare_probe
    assert "readiness_latch" in prepare_probe
    assert "500" in prepare_probe
    assert "'EMBEDDING'" in prepare_probe
    assert "'PRIORITY'" in prepare_probe
    assert "'UPSERT'" in prepare_probe

    assert "AMQPStreamConnection" in assertion_probe
    assert "Topics::EMBEDDING_QUEUE" in assertion_probe
    assert "Topics::PRIORITY_QUEUE" in assertion_probe
    assert "'COMPLETED'" in assertion_probe
    assert "'PUBLISHED'" in assertion_probe
    assert "readiness_latch" in assertion_probe
    assert "hybrid_watermark" in assertion_probe


def test_fixture_proves_product_delete_capture_and_tombstone_processing() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    prepare_probe = (FIXTURE / "prepare-product-delete.php").read_text()
    assertion_probe = (FIXTURE / "assert-product-delete.php").read_text()

    assert "prepare-product-delete.php" in runner
    assert runner.count("queue:consumers:start mageos.opensearch_hybrid.correctness.consumer") == 3
    assert "assert-product-delete.php" in runner
    assert "ProductRepositoryInterface::class" in prepare_probe
    assert "deleteById" in prepare_probe
    assert "isSecureArea" in prepare_probe
    assert "State::class" in prepare_probe
    assert "setAreaCode('global')" in prepare_probe
    assert "DataPlanePublisher::class" in prepare_probe
    assert "'DELETE'" in prepare_probe
    assert "'PENDING'" in prepare_probe
    assert "native_state" in assertion_probe
    assert "hybrid_state" in assertion_probe
    assert "'COMPLETED'" in assertion_probe
    assert "'SUPERSEDED'" in assertion_probe
    assert "mageos_opensearch_hybrid_embedding" in assertion_probe
    assert "'DELETED'" in assertion_probe
    assert "NoSuchEntityException" in assertion_probe


def test_workflow_is_immutable_and_runs_supported_install_and_upgrade_fixtures() -> None:
    workflow = WORKFLOW.read_text()

    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "shivammathur/setup-php@f3e473d116dcccaddc5834248c87452386958240" in workflow
    assert "fixture: fresh-3.4" in workflow
    assert "fixture: upgrade-3.3-to-3.4" in workflow
    assert workflow.count("php: '8.4'") == 2
    assert workflow.count("php: '8.5'") == 1
    assert "php-version: ${{ matrix.php }}" in workflow
    assert "composer validate --strict --no-check-publish" in workflow
    assert "Configure fixture paths" in workflow
    assert "scripts/50_build_module_package.py build" in workflow
    assert "scripts/50_build_module_package.py verify" in workflow
    assert 'unzip -q "${MAGEOS_MODULE_ARCHIVE}"' in workflow
    assert "dev/ci/compose.yaml" in workflow
    assert "dev/ci/run.sh" in workflow
    assert "dev/ci/run-upgrade.sh" in workflow
    assert "permissions:\n  contents: read" in workflow


def test_fixture_installs_the_module_while_upgrading_mageos_33_to_34() -> None:
    runner = (FIXTURE / "run-upgrade.sh").read_text()
    probe = (FIXTURE / "assert-platform-upgrade.php").read_text()

    assert "set -euo pipefail" in runner
    assert "mage-os/project-community-edition" in runner
    assert '"3.3.0"' in runner
    assert "mage-os/module-opensearch-hybrid:@dev" in runner
    assert "mage-os/mageos-async-events:^4.0.8" in runner
    assert "setup:install" in runner
    assert 'assert-platform-upgrade.php" "${fixture_root}" prepare' in runner
    assert "mage-os/product-community-edition=3.4.0" in runner
    assert "require-commerce" in runner
    assert "--force-root-updates" in runner
    assert "--no-update" in runner
    assert runner.count("COMPOSER_MIRROR_PATH_REPOS=1") == 2
    assert (
        'COMPOSER_MIRROR_PATH_REPOS=1 composer --working-dir="${fixture_root}" update'
        in runner
    )
    assert 'assert-package-version.php" "${fixture_root}" "3.4.0"' in runner
    assert "setup:upgrade --keep-generated" in runner
    assert "setup:db:status" in runner
    assert "setup:di:compile" in runner
    assert "module:status MageOS_OpenSearchHybrid" in runner
    assert 'assert-platform-upgrade.php" "${fixture_root}" verify' in runner
    assert "assert-doctor.php" in runner
    assert "composer --working-dir=\"${fixture_root}\" audit" in runner
    assert "CycloneDX:make-sbom" in runner
    assert "dev/sbom/verify.py" in runner
    assert "phpunit.xml.dist" in runner
    assert runner.index("setup:install") < runner.index("mage-os/module-opensearch-hybrid:@dev")
    assert runner.index("mage-os/module-opensearch-hybrid:@dev") < runner.index(
        "setup:upgrade --keep-generated"
    )

    assert "ProductRepositoryInterface::class" in probe
    assert "mageos_opensearch_hybrid_change_journal" in probe
    assert "upgrade_fixture_prepare" in probe
    assert "upgrade_fixture_verify" in probe
    assert "catalog_created_at" in probe


def test_upgrade_runbook_matches_the_verified_mageos_transaction() -> None:
    runbook = UPGRADE_RUNBOOK.read_text()

    assert "mage-os/product-community-edition=<approved-3.4-version>" in runbook
    assert "require-commerce" in runbook
    assert "--force-root-updates" in runbook
    assert "mage-os/module-opensearch-hybrid:<approved-module-version>" in runbook
    assert "mage-os/mageos-async-events:^4.0.8" in runbook
    assert "setup:upgrade --keep-generated" in runbook
    assert "restore the complete pre-change release point" in runbook


def test_workflow_sets_runner_temp_paths_after_the_job_starts() -> None:
    workflow = WORKFLOW.read_text()

    assert "MAGEOS_FIXTURE_ROOT: ${{ runner.temp }}" not in workflow
    assert 'MAGEOS_FIXTURE_ROOT=${RUNNER_TEMP}/mageos-opensearch-hybrid-ci' in workflow
    assert 'MAGEOS_MODULE_ARCHIVE=${RUNNER_TEMP}/mage-os-module-opensearch-hybrid.zip' in workflow
    assert (
        'MAGEOS_MODULE_MANIFEST=${RUNNER_TEMP}/mage-os-module-opensearch-hybrid.manifest.json'
        in workflow
    )
    assert 'MAGEOS_MODULE_INSTALL_ROOT=${RUNNER_TEMP}/mage-os-module-opensearch-hybrid' in workflow
    assert '>> "${GITHUB_ENV}"' in workflow


def test_workflow_runs_the_pinned_semantic_validator() -> None:
    workflow = WORKFLOW.read_text()
    makefile = MAKEFILE.read_text()

    assert "make workflow-check" in workflow
    assert "workflow-check:" in makefile
    assert "github.com/rhysd/actionlint/cmd/actionlint@v1.7.12" in makefile


def test_encoder_check_installs_model_runtime_dependencies() -> None:
    makefile = MAKEFILE.read_text()
    encoder_check = makefile.split("encoder-check:", 1)[1].split("\n\n", 1)[0]

    assert "uv run --frozen --extra models pytest" in encoder_check


def test_fixture_runs_locked_composer_security_audit() -> None:
    runner = (FIXTURE / "run.sh").read_text()

    assert 'composer --working-dir="${fixture_root}" audit' in runner
    assert "--locked" in runner
    assert "--abandoned=ignore" in runner


def test_fixture_enforces_the_module_php_coding_standard() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    upgrade_runner = (FIXTURE / "run-upgrade.sh").read_text()
    ruleset = PHPCS_RULESET.read_text()

    assert '<rule ref="Magento2"/>' in ruleset
    assert '<exclude-pattern>*/dev/*</exclude-pattern>' in ruleset
    assert '<arg name="warning-severity" value="0"/>' in ruleset
    for fixture_runner in (runner, upgrade_runner):
        assert 'vendor/bin/phpcs"' in fixture_runner
        assert 'phpcs.xml.dist"' in fixture_runner


def test_fixture_exercises_bounded_query_openmetrics_after_storefront_requests() -> None:
    runner = (FIXTURE / "run.sh").read_text()

    assert "--query-window=300" in runner
    assert "--query-max-bytes=1048576" in runner
    assert "mageos_opensearch_hybrid_query_log_available 1" in runner
    assert "mageos_opensearch_hybrid_query_window_complete 1" in runner
    assert "generation_id|raw_query|error_class" in runner


def test_fixture_generates_and_retains_a_resolved_module_sbom() -> None:
    workflow = WORKFLOW.read_text()
    runner = (FIXTURE / "run.sh").read_text()
    upgrade_runner = (FIXTURE / "run-upgrade.sh").read_text()
    tool_manifest = json.loads((SBOM_TOOL / "composer.json").read_text())
    tool_lock = json.loads((SBOM_TOOL / "composer.lock").read_text())
    locked_packages = {package["name"]: package["version"] for package in tool_lock["packages"]}

    assert tool_manifest["require"]["cyclonedx/cyclonedx-php-composer"] == "6.2.0"
    assert locked_packages["cyclonedx/cyclonedx-php-composer"] == "v6.2.0"
    assert "MAGEOS_SBOM_TOOL_ROOT=${RUNNER_TEMP}/mage-os-opensearch-hybrid-sbom-tool" in workflow
    assert 'mkdir "${MAGEOS_SBOM_TOOL_ROOT}"' in workflow
    assert "cp dev/sbom/tool/composer.json" in workflow
    assert "dev/sbom/tool/composer.lock" in workflow
    assert '"${MAGEOS_SBOM_TOOL_ROOT}/"' in workflow
    assert 'composer --working-dir="${MAGEOS_SBOM_TOOL_ROOT}" install' in workflow
    assert 'composer --working-dir="${MAGEOS_SBOM_TOOL_ROOT}" audit' in workflow
    assert "COMPOSER_VENDOR_DIR" not in workflow
    assert "MAGEOS_SBOM_PATH=${RUNNER_TEMP}/mage-os-module-opensearch-hybrid" in workflow
    for fixture_runner in (runner, upgrade_runner):
        assert (
            'sbom_tool_root="${MAGEOS_SBOM_TOOL_ROOT:-${fixture_parent}/'
            'mageos-opensearch-hybrid-sbom-tool}"'
            in fixture_runner
        )
        assert '[[ ! -f "${sbom_tool_root}/vendor/autoload.php" ]]' in fixture_runner
        assert (
            'composer --working-dir="${sbom_tool_root}" CycloneDX:make-sbom' in fixture_runner
        )
        assert "COMPOSER_VENDOR_DIR" not in fixture_runner
    assert "CycloneDX:make-sbom" in runner
    assert "--spec-version=1.6" in runner
    assert "--output-reproducible" in runner
    assert "--validate" in runner
    assert "--omit=dev" in runner
    assert "--omit=plugin" in runner
    assert "dev/sbom/verify.py" in runner
    assert "--normalize" in runner
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    assert "retention-days: 30" in workflow


def test_fixture_proves_dependent_invalidation_coalescing_and_resumable_seeding() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    dependent = (FIXTURE / "assert-dependent-invalidation.php").read_text()
    resume = (FIXTURE / "assert-build-resume.php").read_text()

    assert "assert-dependent-invalidation.php" in runner
    assert "assert-build-resume.php" in runner
    assert "CategoryRepositoryInterface" in dependent
    assert "category_name_save" in dependent
    assert "SUPERSEDED" in dependent
    assert "CategoryTextResolver" in dependent
    assert "BuildService::class" in resume
    assert "pauseFake" in resume
    assert "resumeFake" in resume
    assert "'PAUSED'" in resume
    assert "PAUSED_CAPACITY" in resume
    assert "seeding_state" in resume
    assert "seeded_products" in resume
    assert "$configWriter->save(Config::XML_PATH_BATCH_SIZE, '64');" in resume
    assert "$configWriter->save(Config::XML_PATH_MAX_BATCHES, '8');" in resume


def test_fixture_proves_dual_generation_catch_up_and_delete_coverage() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-dual-generation-catch-up.php").read_text()

    assert "assert-dual-generation-catch-up.php" in runner
    assert "->where('generation.state IN (?)', ['BUILDING', 'CATCHING_UP', 'READY'])" in probe
    assert "generation.coverage_complete = generation.coverage_total" in probe
    assert "generation.coverage_failed = ?" in probe
    assert "ValidationService::class" in probe
    assert "resultContractDigest" in probe
    assert "'CATCHING_UP'" in probe
    assert "coverage_complete" in probe
    assert "coverage_total" in probe
    assert "'DELETED'" in probe


def test_fixture_proves_native_and_retained_generation_rollback_without_rebuild() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-admin-activation-controls.php").read_text()

    assert "assert-admin-activation-controls.php" in runner
    assert "previewNativeRollback" in probe
    assert "rollbackToNativeConfirmed" in probe
    assert "previewGenerationRollback" in probe
    assert "rollbackToGenerationConfirmed" in probe
    assert "mageos_opensearch_hybrid_embedding" in probe
    assert "mageos_opensearch_hybrid_outbox" in probe
    assert "$beforeRollbackSet" in probe
    assert "$afterRestoreSet" in probe
    assert "rolled_back_to_generation" in probe


def test_fixture_rehearses_the_state_bound_operator_cli_lifecycle() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-cli-activation-controls.php").read_text()

    assert "assert-cli-activation-controls.php" in runner
    assert "ActivateCommand::class" in probe
    assert "RollbackCommand::class" in probe
    assert "'--dry-run' => true" in probe
    assert "'--confirm' => (string)$activationPreview['confirmation_token']" in probe
    assert "'--confirm' => (string)$nativePreview['confirmation_token']" in probe
    assert "'--confirm' => (string)$generationPreview['confirmation_token']" in probe
    assert "state-bound CLI confirmation" in probe


def test_fixture_rehearses_state_bound_subscription_reconciliation() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-subscription-cli-controls.php").read_text()

    assert "assert-subscription-cli-controls.php" in runner
    assert "EventsInstallCommand::class" in probe
    assert "'--dry-run' => true" in probe
    assert "'--confirm' => (string)$repairPreview['confirmation_token']" in probe
    assert "accepted a stale subscription confirmation" in probe
    assert "cannot be edited directly" in probe
    assert "cannot be deleted directly" in probe


def test_fixture_rehearses_the_state_bound_production_build_gate() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-production-build-gate.php").read_text()

    assert "assert-production-build-gate.php" in runner
    assert "'confirmation_token'" in probe
    assert "buildProductionConfirmed" in probe
    assert "stale production build confirmation" in probe


def test_cleanup_probe_derives_the_preserved_activation_history_count() -> None:
    probe = (FIXTURE / "assert-cleanup-gate.php").read_text()

    assert "$expectedPreservedActivationRows" in probe
    assert (
        "(int)$preview['database']['preserve'][0]['rows'] "
        "!== $expectedPreservedActivationRows"
    ) in probe
    assert "(int)$preview['database']['preserve'][0]['rows'] !== 1" not in probe


def test_fixture_proves_bulk_attribute_and_website_scope_invalidation() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-source-invalidation.php").read_text()

    assert "assert-source-invalidation.php" in runner
    assert "ProductAction::class" in probe
    assert "updateAttributes" in probe
    assert "updateWebsites" in probe
    assert "product_attribute_bulk_update" in probe
    assert "product_website_remove" in probe
    assert "product_website_add" in probe
    assert "'DELETE'" in probe
    assert "'UPSERT'" in probe
    assert "setWebsiteIds([])" in probe
    assert "setWebsiteIds([1])" in probe


def test_fixture_proves_priority_only_inventory_salability_invalidation() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-inventory-invalidation.php").read_text()

    assert "assert-inventory-invalidation.php" in runner
    assert "SourceItemsSaveInterface::class" in probe
    assert "SourceItemInterface::STATUS_IN_STOCK" in probe
    assert "SourceItemInterface::STATUS_OUT_OF_STOCK" in probe
    assert "inventory_salability_change" in probe
    assert "'REFRESH'" in probe
    assert "'EMBEDDING'" in probe
    assert "is_salable" in probe


def test_fixture_proves_tier_price_priority_refresh_without_reencoding() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-price-invalidation.php").read_text()
    dimensions_probe = (FIXTURE / "assert-price-dimensions.php").read_text()

    assert "assert-price-invalidation.php" in runner
    assert "SourceItemsSaveInterface::class" in probe
    assert "SourceItemInterface::STATUS_IN_STOCK" in probe
    assert "TierPriceStorageInterface::class" in probe
    assert "TierPriceInterface::PRICE_TYPE_FIXED" in probe
    assert "setQuantity(1.0)" in probe
    assert "getView()->update()" in probe
    assert "product_price_index" in probe
    assert "customer_group_prices" in probe
    assert "tier_price" in probe
    assert "'REFRESH'" in probe
    assert "'EMBEDDING'" in probe
    assert "assert-price-dimensions.php" in runner
    assert "website_and_customer_group" in runner
    assert "PriceIndexResolver::class" in dimensions_probe
    assert "DimensionModeConfiguration::class" in dimensions_probe
    assert "tier_price" in dimensions_probe


def test_fixture_proves_full_price_and_catalog_rule_invalidation() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-full-price-and-catalog-rule-invalidation.php").read_text()

    assert "assert-full-price-and-catalog-rule-invalidation.php" in runner
    assert "RuleFactory::class" in probe
    assert "catalogrule_rule" in probe
    assert "catalog_product_price" in probe
    assert "product_price_full_reindex" in probe
    assert "product_price_index" in probe
    assert "final_price" in probe
    assert "'REFRESH'" in probe
    assert "'EMBEDDING'" in probe


def test_fixture_proves_store_label_changes_reindex_mapped_attribute_values() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-attribute-option-invalidation.php").read_text()

    assert "assert-attribute-option-invalidation.php" in runner
    assert "ProductAttributeOptionUpdateInterface::class" in probe
    assert "semantic_feature_attributes" in probe
    assert "attribute_option_label_change" in probe
    assert "source_hash" in probe
    assert "embedding_eligible" in probe
    assert "'PRIORITY'" in probe
    assert "'EMBEDDING'" in probe
    assert runner.index("assert-attribute-option-invalidation.php") < runner.index(
        "assert-scope-invalidation.php"
    )


def test_fixture_proves_store_scope_changes_fail_closed_until_rebuilt() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-scope-invalidation.php").read_text()

    assert "assert-scope-invalidation.php" in runner
    assert "StoreRepositoryInterface::class" in probe
    assert "BuildService::class" in probe
    assert "ValidationService::class" in probe
    assert "scope_digest" in probe
    assert "STORE" in probe
    assert "CONFIGURATION" in probe
    assert "readiness_latch" in probe
    assert "'FAILED'" in probe
    assert "acknowledgeGenerationInvalidations" in probe


def test_fixture_proves_bounded_retry_dead_state_and_manual_replay() -> None:
    runner = (FIXTURE / "run.sh").read_text()
    probe = (FIXTURE / "assert-retry-dead-letter.php").read_text()

    assert "assert-retry-dead-letter.php" in runner
    assert "OutboxRepository::class" in probe
    assert "XML_PATH_JOB_MAX_ATTEMPTS" in probe
    assert "'RETRY'" in probe
    assert "'DEAD'" in probe
    assert "prepareManualRetry" in probe
    assert "replay_count" in probe


def test_workflow_validates_prometheus_monitoring_contracts() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "prom/prometheus@sha256:"
        "c62e2f333beae2fdac8c3c7a3daaa7ac81110023cbc5857b0ec541f09d997951"
    ) in workflow
    assert "prometheus-alert-rules.yml" in workflow
    assert "prometheus-alert-rules.test.yml" in workflow
    assert "prometheus-encoder-scrape.yml.example" in workflow
    assert "prometheus-opensearch-scrape.yml.example" in workflow
    assert "prometheus-rabbitmq-scrape.yml.example" in workflow
    assert "/run/secrets/encoder_metrics_token:ro" in workflow
    assert "/run/secrets/opensearch_metrics_password:ro" in workflow
    assert "/run/secrets/opensearch_metrics_ca.pem:ro" in workflow
    assert "/run/secrets/rabbitmq_metrics_password:ro" in workflow
    assert "/run/secrets/rabbitmq_metrics_ca.pem:ro" in workflow
