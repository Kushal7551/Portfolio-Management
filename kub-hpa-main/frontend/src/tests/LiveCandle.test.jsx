import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import LiveCandle from '../Components/Algorithmic-Trading/LiveCandle';

test('renders dropdown and candlestick chart', async () => {
  render(<LiveCandle />);
  expect(screen.getByLabelText('Select Dataset:')).toBeInTheDocument();
  expect(screen.getByText('Live Candle Stick')).toBeInTheDocument();
});

test('fetches and displays CSV data', async () => {
  render(<LiveCandle />);
  fireEvent.change(screen.getByLabelText('Select Dataset:'), { target: { value: 'ADANIPOWER.csv' } });

  await waitFor(() => {
    expect(screen.getByText(/Stock Price/)).toBeInTheDocument();
  });
});
