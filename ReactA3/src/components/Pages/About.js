import React from 'react';
import '../../App.css';
import Card from '../Card';
import Footer from '../Footer';
//import dinal from '../../../public/images/tempmember1.jpg';
//import dinal from './images/tempmember1.jpg';

function About() {
    return (
        <>            
            <h1 className='about'>Meet the team!</h1>
            <h2 className='about'>24 - The AI Merchant</h2>
            <div className='about'>
                <img src='/images/tempmember1.jpg' width={500} height={500} alt="Dinal"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Dinal</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <div className='about'>
                <img src='/images/tempmember1.jpg' width={500} height={500} alt="Dinal"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Dinal</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <div className='about'>
                <img src='/images/tempmember1.jpg' width={500} height={500} alt="Dinal"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Dinal</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <Card />
            <Footer />
        </>
    );
}

export default About;
// export default function About() {
//     return <h1 className='about'>ABOUT</h1>
// }