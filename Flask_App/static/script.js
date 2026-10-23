const form = document.getElementById('valuation-form');
const ticketAmount = document.getElementById('ticket-amount');
const ticketSub = document.getElementById('ticket-sub');
const ticketModel = document.getElementById('ticket-model');
const transportSlider = document.getElementById('transport_score');
const transportVal = document.getElementById('transport_val');
const submitBtn = form.querySelector('.submit-btn');

const floorNoInput = document.getElementById('floor_no');
const totalFloorsInput = document.getElementById('total_floors');

function validateFloors() {
  const floorNo = parseInt(floorNoInput.value, 10);
  const totalFloors = parseInt(totalFloorsInput.value, 10);
  if (!isNaN(floorNo) && !isNaN(totalFloors) && floorNo > totalFloors) {
    floorNoInput.setCustomValidity("Floor number cannot exceed total floors in the building.");
  } else {
    floorNoInput.setCustomValidity("");
  }
}

floorNoInput.addEventListener('input', validateFloors);
totalFloorsInput.addEventListener('input', validateFloors);

// Run check once initially to validate default values
validateFloors();

transportSlider.addEventListener('input', () => {
  transportVal.textContent = transportSlider.value;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Assessing…';
  ticketAmount.style.opacity = 0.4;

  const formData = new FormData(form);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.success) {
      ticketAmount.textContent = data.prediction_formatted;
      ticketSub.textContent = 'Valuation generated from the current survey sheet inputs.';
      ticketModel.textContent = data.model_used;
    } else {
      ticketAmount.textContent = '₹ —';
      ticketSub.textContent = 'Could not generate an estimate: ' + data.error;
    }
  } catch (err) {
    ticketAmount.textContent = '₹ —';
    ticketSub.textContent = 'Network error. Please try again.';
  } finally {
    ticketAmount.style.opacity = 1;
    submitBtn.disabled = false;
    submitBtn.textContent = 'Assess property value';
  }
});
