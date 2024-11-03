import React, { useState } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import Home from './components/Pages/Home';
import About from './components/Pages/About';
import Buy from './components/Pages/Buy';
import Options from './components/Pages/Options';
import Login from './components/Pages/Login';


function App() {
  return (
    <>
    <Router>
      <Navbar />
      <Routes>
        <Route path='/' exact element={ <Home />}></Route>
        <Route path='/About' exact element={ <About />}></Route>
        <Route path='/Buy' exact element={ <Buy />}></Route>
        <Route path='/Options' exact element={ <Options />}></Route>
        <Route path='/Login' exact element={ <Login />}></Route>
      </Routes>
    </Router>
    </>
  )
}

export default App;

// import './App.css';
// import React, { useState } from 'react';
// import { AppBar, Toolbar, Typography, Container, Grid, Card, CardContent, Button, Box, Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton, TextField, Switch, Snackbar, Alert, Fab, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, CircularProgress, LinearProgress, Chip, Avatar, Divider } from '@mui/material';
// import {Menu as MenuIcon,
//         Home as HomeIcon,
//         Info as InfoIcon,
//         Mail as MailIcon,
//         Add as AddIcon,
// } from '@mui/icons-material';

// function App() {
//   const [drawerOpen, setDrawerOpen] = useState(false);

//   const toggleDrawer = (open) => () => {
//     setDrawerOpen(open);
//   };

//   const drawerContent = (
//     <Box sx={{ width: 250 }} role="presentation" onClick={toggleDrawer(false)}>
//       <List>
//         {['Home', 'About', 'Contact'].map((text, index) => (
//           <ListItem button key={text}>
//             <ListItemIcon>{index === 0 ? <HomeIcon /> : index === 1 ? <InfoIcon/> : <MailIcon />}</ListItemIcon>
//             <ListItemText primary={text} />
//           </ListItem>
//         ))}
//       </List>
//     </Box>
//   );

//   return (
//     <Box sx={{flexGrow: 1}}>
//       <AppBar position="static">
//         <Toolbar>
//           <IconButton edge="start" color="inherit" aria-label="menu" onClick={toggleDrawer(true)}>
//             <MenuIcon />
//           </IconButton>
//           <Typography variant="h6" sx={{flexGrow: 1}}>
//             My MUI App
//           </Typography>
//           <Button color="inherit">Contact</Button>
//         </Toolbar>
//       </AppBar>
//       <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer(false)}>
//         {drawerContent}
//       </Drawer>
//     </Box>
//   );
// }

// export default App;
