import React from 'react';
import './Card.css';
import CardItem from './CardItem';

function Cards() {
  return (
    <div className='cards'>
      <h1>Check out these house listings!</h1>
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='ghafjgpaonv'
              label='Adventure'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='fa;k;dlla;mvaf'
              label='Luxury'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='afadgassdddwerw'
              label='Luxury'
              path='/services'
            />
          </ul>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='Set Sail in the Atlantic Ocean visiting Uncharted Waters'
              label='Mystery'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Experience Football on Top of the Himilayan Mountains'
              label='Adventure'
              path='/products'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Ride through the Sahara Desert on a guided camel tour'
              label='Adrenaline'
              path='/sign-up'
            />
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cards;
