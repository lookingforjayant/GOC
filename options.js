// Saves options to chrome.storage
const saveOptions = () => {
    //const color = document.getElementById('color').value;
    //const likesColor = document.getElementById('like').checked;
    const element = document.getElementById('element').value;
    const attribute = document.getElementById('attribute').value;
    const rule = document.getElementById('rule').value;


    chrome.storage.sync.set(
        //{ favoriteColor: color, likesColor: likesColor },
        { element: element, attribute: attribute, rule: rule },
        () => {
            // Update status to let user know options were saved.
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

// Restores select box and checkbox state using the preferences
// stored in chrome.storage.
const restoreOptions = () => {
    chrome.storage.sync.get(
        //{ favoriteColor: 'red', likesColor: true },
        { element: element, attribute: attribute, rule: rule },
        (items) => {
            //document.getElementById('color').value = items.favoriteColor;
            //document.getElementById('like').checked = items.likesColor;
            document.getElementById('element').value = items.element;
            document.getElementById('attribute').value = items.attribute;
            document.getElementById('rule').value = items.rule;
        }
    );
};

document.addEventListener('DOMContentLoaded', restoreOptions);
document.getElementById('save').addEventListener('click', saveOptions);