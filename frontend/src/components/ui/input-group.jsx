import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

function InputGroup({ className, ...props }) {
  return (
    <div
      data-slot="input-group"
      className={cn('flex w-full items-stretch rounded-md', className)}
      {...props}
    />
  )
}

function InputGroupInput({ className, ...props }) {
  return (
    <Input
      data-slot="input-group-input"
      className={cn('rounded-r-none border-r-0 focus-visible:z-10', className)}
      {...props}
    />
  )
}

function InputGroupAddon({ className, align = 'inline-end', ...props }) {
  return (
    <div
      data-slot="input-group-addon"
      data-align={align}
      className={cn(
        'flex items-stretch',
        align === 'inline-start' ? 'order-first' : 'order-last',
        className,
      )}
      {...props}
    />
  )
}

function InputGroupButton({ className, size = 'default', ...props }) {
  return (
    <Button
      data-slot="input-group-button"
      size={size}
      className={cn('rounded-l-none border border-input px-3', className)}
      {...props}
    />
  )
}

export {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
}
