import React, { useState } from 'react';
import './Cards.css';
import CardItem from './CardItem';

function Cards() {
  const [searchQuery, setSearchQuery] = useState('');

  const cardData = [
    { src: 'images/grouplogo.jpg', text: 'ghafjgpaonvgihklj', price: '10k', path: '/services' },
    { src: 'images/grouplogo.jpg', text: 'fa;k;dlla;mvaf', price: 'Luxury', path: '/services' },
    { src: 'images/grouplogo.jpg', text: 'afadgassdddwerw', price: 'Luxury', path: '/services' },
    { src: 'images/grouplogo.jpg', text: 'Set Sail Uncharted Waters', price: 'Mystery', path: '/services' },
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