document.addEventListener('DOMContentLoaded', restoreOptions);
document.getElementById('save').addEventListener('click', saveOptions);

function saveOptions() {
    const element = document.getElementById('element').value;
    const attribute = document.getElementById('attribute').value;
    const rule = document.getElementById('rule').value;

    chrome.storage.sync.set(
        { element: element, attribute: attribute, rule: rule },
        () => {
            const status = document.getElementById('status');
            status.textContent = 'Options saved.';
            setTimeout(() => {
                status.textContent = '';
                document.getElementById('element').value = 0;
                document.getElementById('attribute').value = 0;
                document.getElementById('rule').value = 0;
            }, 1000);
        }
    );
};

// stored in chrome.storage.
function restoreOptions() {
    chrome.storage.sync.get(
        { element: element, attribute: attribute, rule: rule },
        (items) => {
            document.getElementById('element').value = items.element;
            document.getElementById('attribute').value = items.attribute;
            document.getElementById('rule').value = items.rule;
        }
    );
};

