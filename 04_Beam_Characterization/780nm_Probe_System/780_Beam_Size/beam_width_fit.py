import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load your CSV file
file_path = r"C:\Users\ikcco\OneDrive\Desktop\Research Odom\Random Shit To Organize Later\Beam Size\Beam Size Used\Horizontal.csv" # Replace with the actual path to your file
data = pd.read_csv(file_path)

# Rename the columns if needed (assuming first column is x and second column is y)
data.columns = ['x', 'y']

# Define a Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Fit the Gaussian model to the data
x_data = data['x']
y_data = data['y']
initial_guess = [max(y_data), np.mean(x_data), np.std(x_data)]  # Initial guess for the parameters

params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

# Extract the fitted parameters
a_fit, b_fit, c_fit = params

# Generate y-values based on the Gaussian fit for the entire x-axis range (0 to 1023)
x_fit = np.linspace(0, 1023, 1000)
y_fit = gaussian(x_fit, a_fit, b_fit, c_fit)

# Calculate the 1/e^2 points (width of the beam)
x_1e2_left = b_fit - 2 * c_fit
x_1e2_right = b_fit + 2 * c_fit
x_1e2_width = x_1e2_right - x_1e2_left
pixel_length = 5.2
beam_width = x_1e2_width * pixel_length/1000

# Output the equation of the Gaussian and 1/e^2 width
print(f"Gaussian Fit Equation: y = {a_fit:.3f} * exp(-((x - {b_fit:.3f})^2) / (2 * {c_fit:.3f}^2))")
print(f"1/e^2 Width (Pixels): {x_1e2_right - x_1e2_left:.3f}")
print(f"Beam Width (mm): {beam_width:.3f}")
# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'o', label='Original Data', markersize=0.5)

# Plot the Gaussian fit extended over the full range of x
plt.plot(x_fit, y_fit, label='Gaussian Fit', linestyle='--')

# Plot vertical lines at 1/e^2 points
plt.axvline(x=x_1e2_left, color='r', linestyle=':', label=f'1/e^2 width: {beam_width:.2f} mm')
plt.axvline(x=x_1e2_right, color='r', linestyle=':')

# Labeling the axes and adding a title
plt.xlim(0, 1023)
plt.ylim(0, 255)
plt.xlabel('Pixel Number',fontsize = 14)
plt.ylabel('Intensity',fontsize = 14)

# Adding legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
