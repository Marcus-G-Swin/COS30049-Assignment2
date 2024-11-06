import '../../App.css';
import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
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
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

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
  const [chartData, setChartData] = useState(null);
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

      // Prepare data for the chart
      const squareFootages = [1000, 1500, 2000, 2500, 3000];
      const predictions = await Promise.all(
        squareFootages.map(sf =>
          axios.post('http://localhost:8000/predict/', {
            ...requestData,
            square_footage: sf
          }).then(res => res.data.predicted_price)
        )
      );

      // Creating the chart data using the predictions from the backend
      const newChartData = {
        labels: squareFootages,
        datasets: [
          {
            label: 'Predicted Prices',
            data: predictions,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.1
          },
          {
            label: 'Your Prediction',
            data: [{x: parseFloat(squareFootage), y: response.data.predicted_price}],
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            pointRadius: 8,
            pointHoverRadius: 12,
            showLine: false
          }
        ]
      };
      setChartData(newChartData);  // Set the chart data in state
    } catch (err) {
      setError('Error predicting price. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom>
            House Price Predictor
          </Typography>
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                {/* Input fields for each required parameter */}
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
              {chartData && (
                <Box sx={{ mt: 3 }}>
                  <Line 
                    data={chartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: {
                          position: 'top',
                        },
                        title: {
                          display: true,
                          text: 'Price Predictions by Square Footage'
                        }
                      },
                      scales: {
                        x: {
                          type: 'linear',
                          position: 'bottom',
                          title: {
                            display: true,
                            text: 'Square Footage'
                          }
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Predicted Price ($)'
                          }
                        }
                      }
                    }}
                  />
                </Box>
              )}
            </Paper>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default Data;
