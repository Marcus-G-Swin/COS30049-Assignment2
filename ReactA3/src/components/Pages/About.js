import React from 'react';
import '../../App.css';
import Footer from '../Footer';
//import dinal from '../../../public/images/tempmember1.jpg';
//import dinal from './images/tempmember1.jpg';

function About() {
    return (
        <>            
            <h1 className='about'>Meet the team!</h1>
            <h2 className='about'>24 - The AI Merchant</h2>
            <div className='about'>
                <img src='/images/Dinal.jpg' width={425} height={500} alt="Dinal"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Dinal</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <div className='about'>
                <img src='/images/Marcus.jpg' width={425} height={500} alt="Marcus"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Marcus</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <div className='about'>
                <img src='/images/Chi.jpg' width={425} height={500} alt="Chi"></img>
                <div className='aboutcont'>
                    <h3 className='about'>Chi</h3>
                    <p className='about'>Bro is dead</p>
                </div>
            </div>
            <Footer />
        </>
    );
}

export default About;
// export default function About() {
//     return <h1 className='about'>ABOUT</h1>
// }