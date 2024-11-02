import React from 'react';
import {Button} from './Button';
import './Banner.css';
import '../App.css';

function Banner() {
    return (
        <div className='banner-container'>
            <h1>Need a home?</h1>
            <h2>Ask the Merchant!</h2>
            <h2>He got you covered!</h2>
            <div className="banner-btns">
                <Button classname='btns' buttonStyle='btn--outline' buttonSize='btn--large'>Get Started</Button>
                <Button classname='btns' buttonStyle='btn--primary' buttonSize='btn--large'>Login</Button>
            </div>
        </div>
    )
}

export default Banner