import React, { useState } from 'react';
import './Cards.css';
import CardItem from './CardItem';

function Cards() {
  const [searchQuery, setSearchQuery] = useState('');

  const cardData = [
    { src: 'images/prod1.jpg', text: '2 bedroom house at Richmond', price: '2M', path: '/buy' },
    { src: 'images/prod2.jpg', text: '1 bedroom, 1 parking lot house at Melbourne', price: '3M', path: '/buy' },
    { src: 'images/prod3.jpg', text: '2 bedroom house at Abbotsford', price: '2.3M', path: '/buy' },
    { src: 'images/prod4.jpg', text: '1 bedroom house at Fitzroy', price: '1.8M', path: '/buy' },
  ];

  const filteredCards = cardData.filter(card =>
    card.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className='cards'>
      <h1>Check out these house listings!</h1>
      <p>Search:</p>
      <input
        type='text'
        placeholder='Search'
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        className='card-input'
      />
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            {filteredCards.map((card, index) => (
              <CardItem
                key={index}
                src={card.src}
                text={card.text}
                price={card.price}
                path={card.path}
              />
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cards;