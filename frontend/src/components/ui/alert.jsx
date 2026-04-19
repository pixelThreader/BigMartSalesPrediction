import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const alertVariants = cva(
    'relative w-full rounded-lg border px-4 py-3 text-sm [&>svg~*]:pl-7 [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4',
    {
        variants: {
            variant: {
                default: 'bg-background text-foreground',
                destructive:
                    'border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive',
            },
        },
        defaultVariants: {
            variant: 'default',
        },
    },
)

function Alert({ className, variant, ...props }) {
    return (
        <div
            role="alert"
            data-slot="alert"
            className={cn(alertVariants({ variant }), className)}
            {...props}
        />
    )
}

function AlertTitle({ className, ...props }) {
    return <h5 data-slot="alert-title" className={cn('mb-1 font-medium', className)} {...props} />
}

function AlertDescription({ className, ...props }) {
    return (
        <div
            data-slot="alert-description"
            className={cn('text-sm leading-relaxed text-current/90', className)}
            {...props}
        />
    )
}

export { Alert, AlertTitle, AlertDescription }
