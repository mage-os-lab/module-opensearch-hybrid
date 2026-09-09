<?php
declare(strict_types=1);

namespace MageOS\OpenSearchHybrid\Test\Unit\Model\Text;

class DocumentRendererTest extends \PHPUnit\Framework\TestCase
{
    public function testRendersTheFrozenDocumentRecipe(): void
    {
        $renderer = new \MageOS\OpenSearchHybrid\Model\Text\DocumentRenderer();

        self::assertSame(
            'Trail Shoe. Waterproof shoe. footwear. Men / Running. red',
            $renderer->render(
                ' Trail Shoe ',
                ' Waterproof shoe ',
                ['footwear', '', ' Men / Running ', 'red']
            )
        );
    }

    public function testSourceHashBindsTheExactRenderedText(): void
    {
        $renderer = new \MageOS\OpenSearchHybrid\Model\Text\DocumentRenderer();
        $rendered = $renderer->render('Desk', 'Walnut', ['Furniture']);

        self::assertSame(hash('sha256', $rendered), $renderer->sourceHash('Desk', 'Walnut', ['Furniture']));
    }
}
