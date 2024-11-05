import React from 'react';
import './Card.css';
import CardItem from './CardItem';

function Card() {
  return (
    <div className='cards'>
      <h1>Check out these house listings!</h1>
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='ghafjgpaonvgihklj'
              price='10k'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='fa;k;dlla;mvaf'
              price='Luxury'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='afadgassdddwerw'
              price='Luxury'
              path='/services'
            />
          </ul>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='Set Sail Uncharted Waters'
              price='Mystery'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Experience Football on Top of the Himilayan Mountains'
              price='Adventure'
              path='/products'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Ride through the Sahara Desert on a guided camel tour'
              price='Adrenaline'
              path='/sign-up'
            />
          </ul>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='Set Sail in the Atlantic Ocean visiting Uncharted Waters'
              price='Mystery'
              path='/services'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Experience Football on Top of the Himilayan Mountains'
              price='Adventure'
              path='/products'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Ride through the Sahara Desert on a guided camel tour'
              price='Adrenaline'
              path='/sign-up'
            />
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Card;
