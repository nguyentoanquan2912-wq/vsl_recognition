/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - 'info', 'success', 'error'
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Remove the toast after animation completes
    setTimeout(() => {
        if (container.contains(toast)) {
            container.removeChild(toast);
        }
    }, 3000);
}

/**
 * Format a number to have thousands separators
 * @param {number} num 
 * @returns {string}
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Animate counting up to a target number
 * @param {HTMLElement} element 
 * @param {number} target 
 * @param {number} duration 
 */
function animateValue(element, target, duration = 1000) {
    if (!element) return;
    
    let start = 0;
    const increment = target / (duration / 16); // 60fps
    
    function updateCounter() {
        start += increment;
        if (start < target) {
            element.textContent = formatNumber(Math.floor(start));
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = formatNumber(target);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

// Custom Fetch Wrapper for Error Handling
async function apiFetch(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Đã xảy ra lỗi kết nối');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        showToast(error.message, 'error');
        throw error;
    }
}
