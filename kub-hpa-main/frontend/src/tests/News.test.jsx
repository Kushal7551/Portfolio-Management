import { render, screen, waitFor } from '@testing-library/react';
import News from '../Components/Live-News/News';

test('displays loading screen while fetching data', async () => {
  render(<News />);
  expect(screen.getByText('Model is running...')).toBeInTheDocument();
});

test('fetches and displays news data', async () => {
  render(<News />);
  await waitFor(() => {
    expect(screen.getByText('Latest News')).toBeInTheDocument();
  });
});
