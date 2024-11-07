// Data.js

import '../../App.css';
import '../../Data.css';
import Footer from '../Footer';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import {
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Box,
  CircularProgress,
  MenuItem,
  Slider,
  Autocomplete,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Import the suburbs data
import suburbsData from '../../suburb.json'; // Adjust the path if necessary

// Extract the suburbs array from the imported data
const suburbs = suburbsData.suburbs;

console.log('Suburbs:', suburbs); // For debugging purposes

// Registering Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const theme = createTheme({
  palette: {
    primary: {
      main: '#7D267E',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const houseTypes = ['House', 'Apartment', 'Townhouse'];

function Data() {
  const [squareFootage, setSquareFootage] = useState('');
  const [bedrooms, setBedrooms] = useState('');
  const [bathrooms, setBathrooms] = useState('');
  const [garage, setGarage] = useState('');
  const [lotArea, setLotArea] = useState('');
  const [yearBuilt, setYearBuilt] = useState('');
  const [houseType, setHouseType] = useState('');
  const [suburb, setSuburb] = useState(''); // State variable for selected suburb in the form

  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState('');
  const [lineChartData, setLineChartData] = useState(null);
  const [barChartTypeData, setBarChartTypeData] = useState(null);
  const [barChartBedroomsData, setBarChartBedroomsData] = useState(null);
  const [loading, setLoading] = useState(false);

  const [inputErrors, setInputErrors] = useState({}); // Track specific field errors

  // State variables for slider
  const [xAxisRange, setXAxisRange] = useState([0, 0]);
  const [squareFootagesRange, setSquareFootagesRange] = useState([0, 0]);

  // New state variable for chart suburb selection
  const [chartSuburb, setChartSuburb] = useState('');

  // Input validation
  const validateInputs = () => {
    const errors = {};

    if (squareFootage && (isNaN(squareFootage) || squareFootage <= 0)) {
      errors.squareFootage = 'Square footage must be a positive number.';
    }

    if (bedrooms && (isNaN(bedrooms) || bedrooms < 1 || bedrooms > 10)) {
      errors.bedrooms = 'Bedrooms must be a number between 1 and 10.';
    }

    if (bathrooms && (isNaN(bathrooms) || bathrooms < 0 || bathrooms > 10)) {
      errors.bathrooms = 'Bathrooms must be a number between 0 and 10.';
    }

    if (garage && (isNaN(garage) || garage < 0 || garage > 5)) {
      errors.garage = 'Garage spaces must be a number between 0 and 5.';
    }

    if (lotArea && (isNaN(lotArea) || lotArea <= 0)) {
      errors.lotArea = 'Lot area must be a positive number.';
    }

    if (yearBuilt && (isNaN(yearBuilt) || yearBuilt < 1800 || yearBuilt > 2024)) {
      errors.yearBuilt = 'Year built must be between 1800 and 2024.';
    }

    setInputErrors(errors);
    return Object.keys(errors).length === 0; // Return true if no errors
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateInputs()) return; // Stop if validation fails

    setError('');
    setPredictedPrice(null);
    setLoading(true);

    const requestData = {
      ...(squareFootage && { square_footage: parseFloat(squareFootage) }),
      ...(bedrooms && { bedrooms: parseInt(bedrooms) }),
      ...(bathrooms && { bathrooms: parseInt(bathrooms) }),
      ...(garage && { garage: parseInt(garage) }),
      ...(lotArea && { lot_area: parseFloat(lotArea) }),
      ...(yearBuilt && { year_built: parseInt(yearBuilt) }),
      ...(houseType && { house_type: houseType }),
      ...(suburb && { suburb }),
    };

    try {
      const response = await axios.post('http://localhost:8000/predict/', requestData);
      setPredictedPrice(response.data.predicted_price);

      // Generate square footage values over a wider range
      let baseSqFt = parseFloat(squareFootage);
      if (isNaN(baseSqFt)) {
        baseSqFt = 1000; // Default value if none entered
      }

      // Generate square footage values from baseSqFt * 0.5 to baseSqFt * 1.5
      const minSqFt = Math.max(1, Math.floor(baseSqFt * 0.5));
      const maxSqFt = Math.ceil(baseSqFt * 1.5);
      const stepSize = Math.ceil((maxSqFt - minSqFt) / 20); // Adjust step size for reasonable data points
      const squareFootages = [];

      for (let i = minSqFt; i <= maxSqFt; i += stepSize) {
        squareFootages.push(i);
      }

      // Set the square footage range for the slider
      setSquareFootagesRange([minSqFt, maxSqFt]);
      setXAxisRange([minSqFt, maxSqFt]); // Initialize slider to full range

      // Predicted prices for line chart
      const predictedPrices = await Promise.all(
        squareFootages.map((sf) =>
          axios
            .post('http://localhost:8000/predict/', { ...requestData, square_footage: sf })
            .then((res) => res.data.predicted_price)
        )
      );

      const lineChartData = {
        labels: squareFootages,
        datasets: [
          {
            label: 'Predicted Prices',
            data: predictedPrices,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            fill: false,
            tension: 0.1,
            pointRadius: 4,
          },
          {
            label: 'Your Prediction',
            data: [{ x: parseInt(squareFootage), y: response.data.predicted_price }],
            borderColor: 'rgb(250, 165, 18)',
            backgroundColor: 'rgba(255, 208, 17, 0.5)',
            pointRadius: 8,
            pointHoverRadius: 12,
            showLine: false, // Show only the user's prediction
          },
        ],
      };
      setLineChartData(lineChartData);

      // Generate bar charts for the initial chartSuburb (default to suburb from form)
      setChartSuburb(suburb || 'Abbotsford'); // Default suburb if none selected

    } catch (err) {
      setError('Error predicting price. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Function to generate bar charts when chartSuburb changes
  useEffect(() => {
    if (!chartSuburb) return;

    const generateBarCharts = async () => {
      setLoading(true);

      const requestData = {
        ...(squareFootage && { square_footage: parseFloat(squareFootage) }),
        ...(bedrooms && { bedrooms: parseInt(bedrooms) }),
        ...(bathrooms && { bathrooms: parseInt(bathrooms) }),
        ...(garage && { garage: parseInt(garage) }),
        ...(lotArea && { lot_area: parseFloat(lotArea) }),
        ...(yearBuilt && { year_built: parseInt(yearBuilt) }),
        ...(houseType && { house_type: houseType }),
        suburb: chartSuburb, // Use selected suburb for charts
      };

      try {
        // Bar Chart: Predicted Prices by House Type
        const predictionsByType = await Promise.all(
          houseTypes.map((type) =>
            axios
              .post('http://localhost:8000/predict/', { ...requestData, house_type: type })
              .then((res) => res.data.predicted_price)
          )
        );
        const barChartTypeData = {
          labels: houseTypes,
          datasets: [
            {
              label: `Predicted Prices in ${chartSuburb} by House Type`,
              data: predictionsByType,
              backgroundColor: 'rgba(255, 159, 64, 0.5)',
            },
          ],
        };
        setBarChartTypeData(barChartTypeData);

        // Bar Chart: Predicted Prices by Bedrooms
        const bedroomCounts = [1, 2, 3, 4, 5];
        const predictionsBedrooms = await Promise.all(
          bedroomCounts.map((bed) =>
            axios
              .post('http://localhost:8000/predict/', { ...requestData, bedrooms: bed })
              .then((res) => res.data.predicted_price)
          )
        );
        const barChartBedroomsData = {
          labels: bedroomCounts,
          datasets: [
            {
              label: `Predicted Prices in ${chartSuburb} by Bedrooms`,
              data: predictionsBedrooms,
              backgroundColor: 'rgba(54, 162, 235, 0.5)',
            },
          ],
        };
        setBarChartBedroomsData(barChartBedroomsData);
      } catch (err) {
        setError('Error generating charts. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    generateBarCharts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartSuburb]);

  return (
    <div className="datadiv">
      <ThemeProvider className="datadiv" theme={theme}>
        <h1 className="data">House Price Predictor</h1>
        <Container className="datadiv" maxWidth="md">
          <Box sx={{ my: 4 }}>
            <h1 gutterBottom>Enter your dream house details!</h1>
            <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
              <form onSubmit={handleSubmit}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Square Footage"
                      variant="outlined"
                      value={squareFootage}
                      onChange={(e) => setSquareFootage(e.target.value)}
                      error={!!inputErrors.squareFootage}
                      helperText={inputErrors.squareFootage}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Bedrooms"
                      variant="outlined"
                      value={bedrooms}
                      onChange={(e) => setBedrooms(e.target.value)}
                      error={!!inputErrors.bedrooms}
                      helperText={inputErrors.bedrooms}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Bathrooms"
                      variant="outlined"
                      value={bathrooms}
                      onChange={(e) => setBathrooms(e.target.value)}
                      error={!!inputErrors.bathrooms}
                      helperText={inputErrors.bathrooms}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Garage"
                      variant="outlined"
                      value={garage}
                      onChange={(e) => setGarage(e.target.value)}
                      error={!!inputErrors.garage}
                      helperText={inputErrors.garage}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Lot Area"
                      variant="outlined"
                      value={lotArea}
                      onChange={(e) => setLotArea(e.target.value)}
                      error={!!inputErrors.lotArea}
                      helperText={inputErrors.lotArea}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Year Built"
                      variant="outlined"
                      value={yearBuilt}
                      onChange={(e) => setYearBuilt(e.target.value)}
                      error={!!inputErrors.yearBuilt}
                      helperText={inputErrors.yearBuilt}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      select
                      label="House Type"
                      variant="outlined"
                      value={houseType}
                      onChange={(e) => setHouseType(e.target.value)}
                    >
                      {houseTypes.map((type) => (
                        <MenuItem key={type} value={type}>
                          {type}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item xs={6}>
                    <Autocomplete
                      options={suburbs} // Use the suburbs array
                      value={suburb} // Selected suburb in the form
                      onChange={(event, newValue) => setSuburb(newValue)}
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label="Suburb"
                          variant="outlined"
                          fullWidth
                        />
                      )}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      color="primary"
                      fullWidth
                      disabled={loading}
                    >
                      {loading ? <CircularProgress size={24} /> : 'Predict Price'}
                    </Button>
                  </Grid>
                </Grid>
              </form>
            </Paper>
            {error && (
              <Typography color="error" sx={{ mb: 2 }}>
                {error}
              </Typography>
            )}
            {predictedPrice && (
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Predicted Price: ${predictedPrice.toLocaleString()}
                </Typography>
                <Box sx={{ mt: 3 }}>
                  {lineChartData && (
                    <div>
                      <Typography gutterBottom>Adjust Square Footage Range:</Typography>
                      <Slider
                        value={xAxisRange}
                        onChange={(event, newValue) => setXAxisRange(newValue)}
                        valueLabelDisplay="auto"
                        min={squareFootagesRange[0]}
                        max={squareFootagesRange[1]}
                        marks
                      />
                      <Line
                        data={lineChartData}
                        options={{
                          responsive: true,
                          plugins: {
                            legend: { position: 'top' },
                            title: {
                              display: true,
                              text: 'Predicted Prices by Square Footage',
                            },
                          },
                          scales: {
                            x: {
                              type: 'linear',
                              min: xAxisRange[0],
                              max: xAxisRange[1],
                              title: { display: true, text: 'Square Footage' },
                            },
                            y: {
                              title: { display: true, text: 'Price ($)' },
                            },
                          },
                        }}
                      />
                    </div>
                  )}
                  {/* Suburb Selection for Bar Charts */}
                  <Box sx={{ mt: 4 }}>
                    <Typography variant="h6" gutterBottom>
                      Select Suburb for Bar Charts:
                    </Typography>
                    <Autocomplete
                      options={suburbs}
                      value={chartSuburb}
                      onChange={(event, newValue) => setChartSuburb(newValue)}
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label="Suburb"
                          variant="outlined"
                          fullWidth
                        />
                      )}
                    />
                  </Box>
                  {barChartTypeData && (
                    <Bar
                      data={barChartTypeData}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { position: 'top' },
                          title: {
                            display: true,
                            text: `Price Predictions in ${chartSuburb} by House Type`,
                          },
                        },
                        scales: {
                          x: { title: { display: true, text: 'House Type' } },
                          y: { title: { display: true, text: 'Predicted Price ($)' } },
                        },
                      }}
                    />
                  )}
                  {barChartBedroomsData && (
                    <Bar
                      data={barChartBedroomsData}
                      options={{
                        responsive: true,
                        plugins: {
                          legend: { position: 'top' },
                          title: {
                            display: true,
                            text: `Price Predictions in ${chartSuburb} by Bedrooms`,
                          },
                        },
                        scales: {
                          x: { title: { display: true, text: 'Number of Bedrooms' } },
                          y: { title: { display: true, text: 'Predicted Price ($)' } },
                        },
                      }}
                    />
                  )}
                </Box>
              </Paper>
            )}
          </Box>
        </Container>
        <Footer />
      </ThemeProvider>
    </div>
  );
}

export default Data;
