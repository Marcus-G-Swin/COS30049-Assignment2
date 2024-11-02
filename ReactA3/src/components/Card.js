import React from 'react';
import './Card.css';
import CardItem from './CardItem';

function Cards() {
  return (
    <div className='cards'>
      <h1>Check out these EPIC Destinations!</h1>
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='Ejjjjkkj'
              label='cvbnm'
              path='/'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Travel through the Islands of Bali in a Private Cruise'
              label='Luxury'
              path='/'
            />
          </ul>
          <ul className='cards__items'>
            <CardItem
              src='images/grouplogo.jpg'
              text='Ejjjjkkj'
              label='Adventure'
              path='/'
            />
            <CardItem
              src='images/grouplogo.jpg'
              text='Travel through the Islands of Bali in a Private Cruise'
              label='Luxury'
              path='/'
            />
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cards;