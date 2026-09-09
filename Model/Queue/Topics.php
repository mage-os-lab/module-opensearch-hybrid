<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Model\Queue;

class Topics
{
    public const EMBEDDING = 'mageos.opensearch_hybrid.embedding.process.v1';
    public const PRIORITY = 'mageos.opensearch_hybrid.correctness.process.v1';
    public const EMBEDDING_DEAD = 'mageos.opensearch_hybrid.embedding.dead.v1';
    public const PRIORITY_DEAD = 'mageos.opensearch_hybrid.correctness.dead.v1';

    public const EMBEDDING_QUEUE = 'mageos.opensearch_hybrid.embedding';
    public const PRIORITY_QUEUE = 'mageos.opensearch_hybrid.correctness_priority';
    public const EMBEDDING_DEAD_QUEUE = 'mageos.opensearch_hybrid.embedding.dead';
    public const PRIORITY_DEAD_QUEUE = 'mageos.opensearch_hybrid.correctness_priority.dead';
}
