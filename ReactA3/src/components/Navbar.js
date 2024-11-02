import React, {useState, useEffect} from 'react';
import {Link} from 'react-router-dom';
//import {IconButton} from '@mui/material';
import {Button} from './Button';
import './Navbar.css';
import {Menu as MenuIcon,
        Home as HomeIcon,
    } from '@mui/icons-material'

function Navbar() {
    const [click, setClick] = useState(false);
    const [button, setButton] = useState(true);

    const handleClick = () => setClick(!click);
    const closeMobileMenu = () => setClick(false);

    const showButton = () => {
        if(window.innerWidth <= 960) {
            setButton(false)
        } else {
            setButton(true);
        }
    };
 
    useEffect(() => {
        showButton();
        }
    );

    window.addEventListener('resize', showButton);

    return (
        <>
            <nav className='navbar'>
                <div className='navbar-container'>
                    <Link to="/" className='navbar-logo'>
                        AI Merchant
                    </Link>
                    <div className='menu-icon' onClick={handleClick}>
                        <i className={click ? 'fa fa-times' : 'fa fa-bars'} />
                    </div>
                    <ul className={click ? 'nav-menu active' : 'nav-menu'}>
                        <li className='nav-item'>
                            <Link to ='/' className='nav-links' onClick={closeMobileMenu}>
                                Home
                            </Link>
                        </li>
                        <li className='nav-item'>
                            <Link to ='/About' className='nav-links' onClick={closeMobileMenu}>
                                About us
                            </Link>
                        </li>
                        <li className='nav-item'>
                            <Link to ='/Options' className='nav-links' onClick={closeMobileMenu}>
                                More Options
                            </Link>
                        </li>
                        <li>
                            <Link to ='/Login' className='nav-links-mobile' onClick={closeMobileMenu}>
                                Login
                            </Link>
                        </li>
                    </ul>
                    {button && <Button buttonStyle='btn--outline'>LOGIN</Button>}
                </div>
            </nav>  
        </>
    )
}

export default Navbar