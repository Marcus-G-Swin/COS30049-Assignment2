import '../../App.css';
import '../../Data.css'
import Footer from '../Footer';
import React, { useState } from 'react';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { 
  Container, 
  Typography, 
  TextField, 
  Button, 
  Paper, 
  Grid,
  Box,
  CircularProgress,
  MenuItem
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Registering Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// Dropdown options for House Type and Suburb
const houseTypes = ["House", "Apartment", "Townhouse"];
const suburbs = ["Abbotsford", "Richmond", "Melbourne", "Fitzroy"];

function Data() {
  const [squareFootage, setSquareFootage] = useState('');
  const [bedrooms, setBedrooms] = useState('');
  const [bathrooms, setBathrooms] = useState('');
  const [garage, setGarage] = useState('');
  const [lotArea, setLotArea] = useState('');
  const [yearBuilt, setYearBuilt] = useState('');
  const [houseType, setHouseType] = useState('');
  const [suburb, setSuburb] = useState('');
  
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState('');
  const [lineChartData, setLineChartData] = useState(null);
  const [barChartTypeData, setBarChartTypeData] = useState(null);
  const [barChartBedroomsData, setBarChartBedroomsData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
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
      ...(suburb && { suburb })
    };

    try {
      const response = await axios.post('http://localhost:8000/predict/', requestData);
      setPredictedPrice(response.data.predicted_price);

      // 1. Line Chart: Actual vs Predicted Prices by Square Footage
      // Generate square footage values around the entered value
      let baseSqFt = parseFloat(squareFootage);
      if (isNaN(baseSqFt)) {
        baseSqFt = 1000; // Default value if none entered
      }

      // Determine the step size based on the magnitude
      let stepSize = Math.pow(10, Math.floor(Math.log10(baseSqFt)) - 1);
      if (stepSize < 1) stepSize = 1;

      const squareFootages = [];
      for (let i = baseSqFt - stepSize * 2; i <= baseSqFt + stepSize * 2; i += stepSize) {
        if (i > 0) { // Ensure positive square footage
          squareFootages.push(Math.round(i));
        }
      }

      // Simulated actual prices (for demonstration purposes)
      const actualPrices = squareFootages.map(sqft => {
        // This is a placeholder. Replace with actual data if available.
        return sqft * 5000; // Assuming $5,000 per square foot as an example
      });

      const predictedPrices = await Promise.all(
        squareFootages.map(sf => 
          axios.post('http://localhost:8000/predict/', {...requestData, square_footage: sf})
            .then(res => res.data.predicted_price)
        )
      );

      const lineChartData = {
        labels: squareFootages,
        datasets: [
          {
            label: 'Actual Prices',
            data: actualPrices,
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            fill: false,
            tension: 0.1,
            pointRadius: 4
          },
          {
            label: 'Predicted Prices',
            data: predictedPrices,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            fill: false,
            tension: 0.1,
            pointRadius: 4
          },
          {
            label: 'Your Prediction',
            data: [{x: parseInt(squareFootage), y: response.data.predicted_price}],
            borderColor: 'rgb(250, 165, 18)',
            backgroundColor: 'rgba(255, 208, 17, 0.5)',
            pointRadius: 8,
            pointHoverRadius: 12,
            showLine: false // Show only the user's prediction
          } 
        ]
      };
      setLineChartData(lineChartData);

      // 2. Bar Chart: Predicted Prices by House Type
      const predictionsByType = await Promise.all(
        houseTypes.map(type => 
          axios.post('http://localhost:8000/predict/', {...requestData, house_type: type})
            .then(res => res.data.predicted_price)
        )
      );
      const barChartTypeData = {
        labels: houseTypes,
        datasets: [
          {
            label: 'Predicted Prices by House Type',
            data: predictionsByType,
            backgroundColor: 'rgba(255, 159, 64, 0.5)',
          }
        ]
      };
      setBarChartTypeData(barChartTypeData);

      // 3. Bar Chart: Predicted Prices by Bedrooms
      const bedroomCounts = [1, 2, 3, 4, 5];
      const predictionsBedrooms = await Promise.all(
        bedroomCounts.map(bed => 
          axios.post('http://localhost:8000/predict/', {...requestData, bedrooms: bed})
            .then(res => res.data.predicted_price)
        )
      );
      const barChartBedroomsData = {
        labels: bedroomCounts,
        datasets: [
          {
            label: 'Predicted Prices by Bedrooms',
            data: predictionsBedrooms,
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
          }
        ]
      };
      setBarChartBedroomsData(barChartBedroomsData);

    } catch (err) {
      setError('Error predicting price. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider className='data' theme={theme}>
      <h1 className='data'>House Price Predictor</h1>
      <Container maxWidth="md">
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
                  <TextField
                    fullWidth
                    select
                    label="Suburb"
                    variant="outlined"
                    value={suburb}
                    onChange={(e) => setSuburb(e.target.value)}
                  >
                    {suburbs.map((sub) => (
                      <MenuItem key={sub} value={sub}>
                        {sub}
                      </MenuItem>
                    ))}
                  </TextField>
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
                  <Line 
                    data={lineChartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Actual vs Predicted Prices by Square Footage' }
                      },
                      scales: { 
                        x: { 
                          title: { display: true, text: 'Square Footage' }
                        }, 
                        y: { 
                          title: { display: true, text: 'Price ($)' }
                        } 
                      }
                    }}
                  />
                )}
                {barChartTypeData && (
                  <Bar 
                    data={barChartTypeData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Price Predictions by House Type' }
                      },
                      scales: { 
                        x: { title: { display: true, text: 'House Type' }}, 
                        y: { title: { display: true, text: 'Predicted Price ($)' }} 
                      }
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
                        title: { display: true, text: 'Price Predictions by Bedrooms' }
                      },
                      scales: { 
                        x: { title: { display: true, text: 'Number of Bedrooms' }}, 
                        y: { title: { display: true, text: 'Predicted Price ($)' }} 
                      }
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
  );
}

export default Data;
