export function debounce(func: Function, timeout = 300){
  let timer: number;
  return (...args: any[]) => {
    clearTimeout(timer);
    timer = setTimeout(() => { // @ts-ignore
      func(args); }, timeout) as unknown as number;
  };
}