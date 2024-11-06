import React from 'react';
import {Button1} from '../Button';
import '../../App.css';
import Cards from '../Cards';
import Footer from '../Footer';

function Buy() {
    return (
        <>  
            <div className='buy'>
                <h1 className='buy'>Find a home!</h1>
                <h2 className='buy'>Or want to see the prediction?</h2>
                <div className="buy-btns">
                    <Button1 classname='btns' buttonStyle='btn--outline' buttonSize='btn--large'>Predict</Button1>
                </div>
            </div>          

            <Cards />
            <Footer />
        </>
    );
}

export default Buy;
