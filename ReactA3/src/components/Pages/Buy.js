import React from 'react';
import '../../App.css';
import Cards from '../Cards';
import Footer from '../Footer';

function Buy() {
    return (
        <>  
            <div className='buy'>
                <h1 className='buy'>Give us money!!!!!!</h1>
                <h2 className='buy'>Or did you want to look at the data?</h2>
            </div>          

            <Cards />
            <Footer />
        </>
    );
}

export default Buy;
