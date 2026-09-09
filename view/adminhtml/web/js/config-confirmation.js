define(['jquery', 'mage/translate'], function ($, $t) {
    'use strict';

    return function (config, element) {
        var panel = $(element),
            form = panel.closest('form'),
            impact = panel.find('[data-role="hybrid-config-impact"]'),
            token = panel.find('[data-role="hybrid-config-token"]'),
            confirmation = panel.find('[data-role="hybrid-config-confirmation"]'),
            human = panel.find('[data-role="hybrid-config-human"]');

        function clearPreview() {
            token.val('');
            human.val('').removeClass('required-entry');
            confirmation.prop('hidden', true);
            impact.prop('hidden', true).removeClass('message-success success message-error error').text('');
        }

        form.on('change', '[name^="groups["]', clearPreview);
        panel.find('[data-role="hybrid-config-preview"]').on('click', function () {
            clearPreview();
            $.ajax({
                url: config.previewUrl,
                type: 'POST',
                dataType: 'json',
                data: form.serialize()
            }).done(function (response) {
                var preview = response.preview || {},
                    paths,
                    stores;

                impact.prop('hidden', false).addClass('message-success success');
                if (!preview.requires_confirmation) {
                    impact.text($t('No changed generation-affecting value has an existing hybrid generation.'));
                    return;
                }
                paths = preview.changes.map(function (change) {
                    return change.path;
                }).join(', ');
                stores = preview.stores.map(function (store) {
                    return store.store_id + ': ' + store.estimated_documents;
                }).join(', ');
                impact.text($t('Changes: %1. Store IDs and estimated documents: %2. Total documents: %3.')
                    .replace('%1', paths)
                    .replace('%2', stores)
                    .replace('%3', preview.estimated_documents));
                token.val(preview.confirmation_token);
                human.addClass('required-entry');
                confirmation.prop('hidden', false);
                human.trigger('focus');
            }).fail(function (xhr) {
                var response = xhr.responseJSON || {};

                impact.prop('hidden', false).addClass('message-error error')
                    .text(response.message || $t('The configuration impact preview failed.'));
            });
        });
    };
});
